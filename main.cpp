/*
 * Redressement (orthorectification) d'une page photographiee
 * ----------------------------------------------------------
 * Etapes :
 *   1. Lecture + redimensionnement de travail
 *   2. Detection automatique des 4 coins : les DEUX chaines sont executees
 *        - methode A : fond simple    -> Otsu + morphologie
 *        - methode B : fond texture   -> segmentation couleur (V - S) + median + morphologie
 *                                        (repli : Canny)
 *      Chaque quadrilatere est note par le soutien des contours (gradient de Sobel le long
 *      de ses 4 cotes) ; la methode retenue est la mieux notee (A en cas d'egalite).
 *   3. Affichage du resultat de la detection
 *   4. Ajustement interactif des coins a la souris
 *   5. Transformation de perspective (sur l'image ORIGINALE pleine resolution)
 *
 * Utilisation :
 *   redressement.exe ../images/receipt.jpg            (mode auto)
 *   redressement.exe ../images/receipt.jpg simple
 *   redressement.exe ../images/cell_pic.jpg texture
 *
 * Touches (fenetre d'ajustement) :
 *   Souris : cliquer-glisser un coin pour le deplacer
 *   ENTREE ou 'v' : valider   'r' : reset   '+' / '-' : taille de la fenetre   ECHAP : quitter
 *
 * Compilation : activer C++11 (Code::Blocks : Settings > Compiler > "Have g++ follow C++11")
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

using namespace cv;
using namespace std;

/* ------------------------------------------------------------------ */
/*  Outils                                                            */
/* ------------------------------------------------------------------ */

// Ordonne 4 points : 0=haut-gauche, 1=haut-droit, 2=bas-droit, 3=bas-gauche
static vector<Point2f> ordonner(const vector<Point2f>& p)
{
    vector<Point2f> r(4);
    int iTL = 0, iBR = 0, iTR = 0, iBL = 0;
    for (int i = 1; i < 4; ++i) {
        if (p[i].x + p[i].y < p[iTL].x + p[iTL].y) iTL = i;   // somme minimale
        if (p[i].x + p[i].y > p[iBR].x + p[iBR].y) iBR = i;   // somme maximale
        if (p[i].y - p[i].x < p[iTR].y - p[iTR].x) iTR = i;   // difference minimale
        if (p[i].y - p[i].x > p[iBL].y - p[iBL].x) iBL = i;   // difference maximale
    }
    r[0] = p[iTL]; r[1] = p[iTR]; r[2] = p[iBR]; r[3] = p[iBL];
    return r;
}

// Taille de noyau impaire proportionnelle a la taille de l'image
static int noyauImpair(const Mat& img, double ratio, int minimum = 3)
{
    int k = (int)(min(img.rows, img.cols) * ratio);
    k = max(k, minimum);
    return k | 1;
}

// Proportion de pixels blancs dans une bande de 5% le long des bords
static double proportionBlancBord(const Mat& bin)
{
    int b = max(2, min(bin.rows, bin.cols) / 20);
    Mat m(bin.size(), CV_8UC1, Scalar(255));
    rectangle(m, Rect(b, b, bin.cols - 2 * b, bin.rows - 2 * b), Scalar(0), -1);
    return mean(bin, m)[0] / 255.0;
}

// Soutien des contours : mediane du gradient de Sobel (lisse) le long des 4 cotes du quadrilatere.
// Un vrai bord de page donne une valeur elevee ; un quadrilatere qui coupe le fond en donne une faible.
static double soutienContours(const Mat& bgr, const vector<Point2f>& q)
{
    Mat g, gx, gy, mag;
    cvtColor(bgr, g, COLOR_BGR2GRAY);
    GaussianBlur(g, g, Size(5, 5), 0);
    Sobel(g, gx, CV_32F, 1, 0);
    Sobel(g, gy, CV_32F, 0, 1);
    magnitude(gx, gy, mag);
    dilate(mag, mag, Mat::ones(9, 9, CV_8U));            // tolerance de +/- 4 pixels

    vector<float> v;
    for (int i = 0; i < 4; ++i) {
        Point2f a = q[i], b = q[(i + 1) % 4];
        for (int t = 0; t < 60; ++t) {
            Point2f p = a + (b - a) * (0.05f + 0.9f * t / 59.f);
            int x = cvRound(p.x), y = cvRound(p.y);
            bool dedans = (x >= 0 && y >= 0 && x < mag.cols && y < mag.rows);
            v.push_back(dedans ? mag.at<float>(y, x) : 0.f);
        }
    }
    nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

/* ------------------------------------------------------------------ */
/*  Extraction du quadrilatere a partir d'un masque binaire           */
/* ------------------------------------------------------------------ */
static bool quadDepuisMasque(const Mat& masque, vector<Point2f>& out, double ratioAireMin = 0.05)
{
    vector<vector<Point> > cnts;
    findContours(masque.clone(), cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (cnts.empty()) return false;

    // Plus grande composante connexe = la page
    int best = -1; double bestA = 0;
    for (size_t i = 0; i < cnts.size(); ++i) {
        double a = contourArea(cnts[i]);
        if (a > bestA) { bestA = a; best = (int)i; }
    }
    if (best < 0 || bestA < ratioAireMin * (double)masque.total()) return false;

    // Une feuille est convexe -> on travaille sur l'enveloppe convexe
    vector<Point> hull, approx;
    convexHull(cnts[best], hull);
    double peri = arcLength(hull, true);

    // On augmente progressivement epsilon jusqu'a obtenir <= 4 sommets
    for (double eps = 0.01; eps <= 0.15; eps += 0.005) {
        approxPolyDP(hull, approx, eps * peri, true);
        if (approx.size() <= 4) break;
    }
    if (approx.size() == 4) {
        out.clear();
        for (size_t i = 0; i < 4; ++i) out.push_back(Point2f((float)approx[i].x, (float)approx[i].y));
        return true;
    }
    // Repli : rectangle englobant d'aire minimale
    RotatedRect rr = minAreaRect(hull);
    Point2f v[4]; rr.points(v);
    out.assign(v, v + 4);
    return true;
}

/* ------------------------------------------------------------------ */
/*  Methode A : fond simple (uniforme et contraste)                   */
/* ------------------------------------------------------------------ */
static bool detecterSimple(const Mat& bgr, vector<Point2f>& out, Mat& masque)
{
    Mat gray, flou, bin;
    cvtColor(bgr, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, flou, Size(5, 5), 0);

    // Histogramme bimodal -> seuil automatique d'Otsu
    threshold(flou, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // La page doit etre blanche : si les bords de l'image sont blancs, on inverse
    if (proportionBlancBord(bin) > 0.5) bitwise_not(bin, bin);

    // Nettoyage : fermeture (comble le texte) puis ouverture (retire les petits objets)
    int k = noyauImpair(bgr, 0.02, 5);
    Mat B = getStructuringElement(MORPH_ELLIPSE, Size(k, k));
    morphologyEx(bin, bin, MORPH_CLOSE, B);
    morphologyEx(bin, bin, MORPH_OPEN, B);

    masque = bin.clone();
    return quadDepuisMasque(bin, out);
}

/* ------------------------------------------------------------------ */
/*  Methode B : fond texture (bois, bruit, grain...)                  */
/* ------------------------------------------------------------------ */
static bool detecterTexture(const Mat& bgr, vector<Point2f>& out, Mat& masque)
{
    // 1) Critere colorimetrique : papier = tres peu sature et lumineux
    //    score = V - S  (eleve pour le papier, faible pour le bois sature)
    Mat hsv; cvtColor(bgr, hsv, COLOR_BGR2HSV);
    vector<Mat> ch; split(hsv, ch);
    Mat score;
    addWeighted(ch[2], 1.0, ch[1], -1.0, 0.0, score);   // saturation arithmetique 8 bits

    // 2) Le filtre median ecrase le grain du bois sans detruire les bords de la page
    medianBlur(score, score, noyauImpair(bgr, 0.012, 5));

    // 3) Seuillage automatique
    Mat bin;
    threshold(score, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
    if (proportionBlancBord(bin) > 0.5) bitwise_not(bin, bin);

    // 4) Morphologie avec un element assez grand : ouverture = elimine les taches du fond,
    //    fermeture = comble le texte et les plis de la page
    int k = noyauImpair(bgr, 0.025, 7);
    Mat B = getStructuringElement(MORPH_ELLIPSE, Size(k, k));
    morphologyEx(bin, bin, MORPH_OPEN, B);
    morphologyEx(bin, bin, MORPH_CLOSE, B);

    masque = bin.clone();
    if (quadDepuisMasque(bin, out, 0.08)) return true;

    // 5) Repli : contours de Canny (seuils adaptatifs) + fermeture
    Mat g, e;
    cvtColor(bgr, g, COLOR_BGR2GRAY);
    medianBlur(g, g, 7);
    double m = mean(g)[0];
    Canny(g, e, 0.66 * m, 1.33 * m);
    dilate(e, e, B);
    morphologyEx(e, e, MORPH_CLOSE, B);
    masque = e.clone();
    return quadDepuisMasque(e, out, 0.08);
}

/* ------------------------------------------------------------------ */
/*  Detection : les deux chaines sont evaluees, la mieux notee est retenue */
/* ------------------------------------------------------------------ */
struct Detection {
    vector<Point2f> pts;       // 4 coins ordonnes (image de travail)
    Mat masqueS, masqueT;      // masques des deux chaines
    double scoreS, scoreT;     // soutien des contours (0 = chaine en echec)
    bool texture;              // chaine retenue
    Detection() : scoreS(0), scoreT(0), texture(false) {}
};

static Detection detecter(const Mat& bgr, const string& mode)
{
    Detection d;
    vector<Point2f> qs, qt;
    bool oks = detecterSimple(bgr, qs, d.masqueS);
    bool okt = detecterTexture(bgr, qt, d.masqueT);
    if (oks) { qs = ordonner(qs); d.scoreS = soutienContours(bgr, qs); }
    if (okt) { qt = ordonner(qt); d.scoreT = soutienContours(bgr, qt); }

    if (!oks && !okt) {        // dernier recours : rectangle centre, a ajuster a la main
        cout << "Detection echouee : rectangle par defaut (a ajuster a la main)\n";
        float mx = bgr.cols * 0.1f, my = bgr.rows * 0.1f;
        d.pts.push_back(Point2f(mx, my));
        d.pts.push_back(Point2f(bgr.cols - mx, my));
        d.pts.push_back(Point2f(bgr.cols - mx, bgr.rows - my));
        d.pts.push_back(Point2f(mx, bgr.rows - my));
        return d;
    }
    if (mode == "simple")       d.texture = !oks;
    else if (mode == "texture") d.texture = okt;
    else                        d.texture = okt && (!oks || d.scoreT > 1.25 * d.scoreS);  // A prefere en cas d'egalite
    d.pts = d.texture ? qt : qs;
    return d;
}

/* ------------------------------------------------------------------ */
/*  Ajustement interactif                                             */
/* ------------------------------------------------------------------ */
struct Etat {
    Mat base;                 // image de travail (sans dessin)
    vector<Point2f> pts;      // coins courants, en coordonnees de l'image de travail
    int sel;                  // indice du coin selectionne (-1 = aucun)
    double f;                 // facteur d'affichage : ecran = travail * f
    string fenetre;
    Etat() : sel(-1), f(1.0) {}
};

static void dessiner(Etat& e)
{
    Mat d;
    resize(e.base, d, Size(), e.f, e.f, e.f < 1.0 ? INTER_AREA : INTER_LINEAR);
    vector<Point2f> p(4);
    for (int i = 0; i < 4; ++i) p[i] = e.pts[i] * (float)e.f;
    for (int i = 0; i < 4; ++i)
        line(d, p[i], p[(i + 1) % 4], Scalar(0, 255, 0), 2, LINE_AA);
    for (int i = 0; i < 4; ++i) {
        circle(d, p[i], 8, Scalar(0, 0, 255), -1, LINE_AA);
        putText(d, to_string(i + 1), p[i] + Point2f(12, -12),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2, LINE_AA);
    }
    putText(d, "Glisser les coins | ENTREE: valider | r: reset | +/-: taille | ECHAP: quitter",
            Point(8, 20), FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 255, 255), 1, LINE_AA);
    imshow(e.fenetre, d);
}

static void onMouse(int event, int x, int y, int, void* ud)
{
    Etat* e = (Etat*)ud;
    Point2f ecran((float)x, (float)y);
    if (event == EVENT_LBUTTONDOWN) {
        double dmin = 30.0; e->sel = -1;                   // rayon de capture : 30 px ecran
        for (int i = 0; i < 4; ++i) {
            double d = norm(e->pts[i] * (float)e->f - ecran);
            if (d < dmin) { dmin = d; e->sel = i; }
        }
    } else if (event == EVENT_MOUSEMOVE && e->sel >= 0) {
        Point2f p = ecran * (float)(1.0 / e->f);           // retour aux coordonnees de travail
        p.x = min(max(p.x, 0.f), (float)(e->base.cols - 1));
        p.y = min(max(p.y, 0.f), (float)(e->base.rows - 1));
        e->pts[e->sel] = p;
        dessiner(*e);
    } else if (event == EVENT_LBUTTONUP) {
        e->sel = -1;
    }
}

/* ------------------------------------------------------------------ */
/*  Transformation de perspective                                     */
/* ------------------------------------------------------------------ */
static Mat redresser(const Mat& orig, const vector<Point2f>& ptsTravail, double echelle)
{
    vector<Point2f> coins = ordonner(ptsTravail);
    vector<Point2f> src(4);
    for (int i = 0; i < 4; ++i) src[i] = coins[i] * (float)(1.0 / echelle);   // retour pleine resolution

    float largeur = (float)max(norm(src[1] - src[0]), norm(src[2] - src[3]));
    float hauteur = (float)max(norm(src[3] - src[0]), norm(src[2] - src[1]));

    vector<Point2f> dst(4);
    dst[0] = Point2f(0, 0);
    dst[1] = Point2f(largeur - 1, 0);
    dst[2] = Point2f(largeur - 1, hauteur - 1);
    dst[3] = Point2f(0, hauteur - 1);

    Mat M = getPerspectiveTransform(src, dst);
    Mat resultat;
    warpPerspective(orig, resultat, M, Size(cvRound(largeur), cvRound(hauteur)), INTER_CUBIC);
    return resultat;
}

/* ------------------------------------------------------------------ */
/*  Affichage adapte a l'ecran                                        */
/* ------------------------------------------------------------------ */
static Mat pourAffichage(const Mat& img, int hMax = 560)
{
    if (img.rows <= hMax) return img;
    double s = (double)hMax / img.rows;
    Mat r; resize(img, r, Size(), s, s, INTER_AREA);
    return r;
}

/* ------------------------------------------------------------------ */
/*  Nettoyage d'un chemin saisi (guillemets, espaces, retour chariot)  */
/* ------------------------------------------------------------------ */
static string nettoyerChemin(string c)
{
    const string blancs = " \t\r\n\"'";
    size_t a = c.find_first_not_of(blancs);
    if (a == string::npos) return "";
    size_t b = c.find_last_not_of(blancs);
    return c.substr(a, b - a + 1);
}

/* ------------------------------------------------------------------ */
/*  Generation des figures du rapport                                 */
/*  Utilisation : redressement.exe figures receipt.jpg cell_pic.jpg desk.JPG   */
/*  (les 3 images dans CET ordre) -> fichiers fig*.png dans le dossier d'execution */
/* ------------------------------------------------------------------ */
static Mat etiqueter(const Mat& img, const string& txt, int hauteur = 420)
{
    Mat bgr;
    if (img.channels() == 1) cvtColor(img, bgr, COLOR_GRAY2BGR); else bgr = img.clone();
    double k = (double)hauteur / bgr.rows;
    Mat r; resize(bgr, r, Size(), k, k, INTER_AREA);
    Mat bandeau(30, r.cols, CV_8UC3, Scalar(255, 255, 255));
    putText(bandeau, txt, Point(6, 21), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 0, 0), 1, LINE_AA);
    Mat out; vconcat(bandeau, r, out);
    copyMakeBorder(out, out, 2, 2, 2, 2, BORDER_CONSTANT, Scalar(128, 128, 128));
    return out;
}

static Mat tracerHistogramme(const Mat& gray, double seuil, int hauteur)
{
    int bins = 256, canal = 0; float plage[] = {0, 256}; const float* pl = plage;
    Mat hist; calcHist(&gray, 1, &canal, Mat(), hist, 1, &bins, &pl);
    double mx; minMaxLoc(hist, 0, &mx);
    Mat img(hauteur, 512, CV_8UC3, Scalar(255, 255, 255));
    for (int i = 0; i < 256; ++i) {
        int v = cvRound(hist.at<float>(i) / mx * (hauteur - 30));
        rectangle(img, Point(2 * i, hauteur - 1), Point(2 * i + 1, hauteur - 1 - v), Scalar(90, 90, 90), -1);
    }
    int x = cvRound(seuil) * 2;
    line(img, Point(x, 0), Point(x, hauteur - 1), Scalar(0, 0, 255), 2);
    putText(img, "seuil d'Otsu = " + to_string(cvRound(seuil)), Point(8, 22),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2, LINE_AA);
    return img;
}

static void polygone(Mat& img, const vector<Point2f>& q, Scalar c, int ep, bool numeroter)
{
    for (int i = 0; i < 4; ++i) line(img, q[i], q[(i + 1) % 4], c, ep, LINE_AA);
    if (numeroter)
        for (int i = 0; i < 4; ++i) {
            circle(img, q[i], 9, Scalar(0, 0, 255), -1, LINE_AA);
            putText(img, to_string(i + 1), q[i] + Point2f(12, -12), FONT_HERSHEY_SIMPLEX, 0.9,
                    Scalar(255, 0, 0), 2, LINE_AA);
        }
}

static int genererFigures(int n, char** v)
{
    if (n < 3) { cerr << "Usage : figures receipt.jpg cell_pic.jpg desk.JPG" << endl; return -1; }
    const char* noms[3] = {"ticket", "cell_pic", "desk"};
    Mat orig[3], work[3]; double ech[3]; Detection det[3];
    for (int i = 0; i < 3; ++i) {
        orig[i] = imread(nettoyerChemin(v[i]), IMREAD_COLOR);
        if (orig[i].empty()) { cerr << "Impossible de lire : " << v[i] << endl; return -1; }
        ech[i] = min(1.0, 900.0 / max(orig[i].cols, orig[i].rows));
        resize(orig[i], work[i], Size(), ech[i], ech[i], INTER_AREA);
        det[i] = detecter(work[i], "auto");
    }

    // Figure 1 : images originales
    {
        Mat f; hconcat(vector<Mat>{etiqueter(orig[0], "receipt.jpg"), etiqueter(orig[1], "cell_pic.jpg"),
                                   etiqueter(orig[2], "desk.JPG")}, f);
        imwrite("fig01_images_originales.png", f);
    }

    // Figure 2 : ticket, histogramme, Otsu
    {
        Mat g, flou, bin; cvtColor(work[0], g, COLOR_BGR2GRAY);
        GaussianBlur(g, flou, Size(5, 5), 0);
        double t = threshold(flou, bin, 0, 255, THRESH_BINARY | THRESH_OTSU);
        Mat f; hconcat(vector<Mat>{etiqueter(g, "niveaux de gris"),
                                   etiqueter(tracerHistogramme(flou, t, 420), "histogramme"),
                                   etiqueter(bin, "seuillage d'Otsu")}, f);
        imwrite("fig02_ticket_histogramme_otsu.png", f);
    }

    // Figure 3 : masque du ticket (chaine simple)
    imwrite("fig03_masque_ticket.png", etiqueter(det[0].masqueS, "masque - chaine fond simple (ticket)", 520));

    // Figure 4 : echec de la chaine simple sur desk
    {
        vector<Point2f> qs, qt; Mat ms, mt;
        detecterSimple(work[2], qs, ms); detecterTexture(work[2], qt, mt);
        qs = ordonner(qs); qt = ordonner(qt);
        Mat vis = work[2].clone();
        polygone(vis, qs, Scalar(0, 0, 255), 3, false);     // rouge : chaine simple (erronee)
        polygone(vis, qt, Scalar(0, 200, 0), 3, false);     // vert  : chaine texturee
        Mat f; hconcat(vector<Mat>{etiqueter(ms, "masque - chaine simple"),
                                   etiqueter(vis, "rouge: simple | vert: texturee"),
                                   etiqueter(mt, "masque - chaine texturee")}, f);
        imwrite("fig04_echec_chaine_simple_desk.png", f);
    }

    // Figure 5 : canaux H, S, V et score V - S (cell_pic)
    {
        Mat hsv; cvtColor(work[1], hsv, COLOR_BGR2HSV);
        vector<Mat> ch; split(hsv, ch);
        Mat score; subtract(ch[2], ch[1], score);
        Mat f; hconcat(vector<Mat>{etiqueter(work[1], "image"), etiqueter(ch[0], "H (teinte)"),
                                   etiqueter(ch[1], "S (saturation)"), etiqueter(ch[2], "V (valeur)"),
                                   etiqueter(score, "score V - S")}, f);
        imwrite("fig05_canaux_HSV_score.png", f);
    }

    // Figure 6 : masques de la chaine texturee (cell_pic et desk)
    {
        Mat f; hconcat(vector<Mat>{etiqueter(det[1].masqueT, "cell_pic.jpg - masque (chaine texturee)"),
                                   etiqueter(det[2].masqueT, "desk.JPG - masque (chaine texturee)")}, f);
        imwrite("fig06_masques_texture.png", f);
    }

    // Figure 7 : contour, enveloppe convexe, quadrilatere (desk)
    {
        vector<vector<Point> > cnts;
        findContours(det[2].masqueT.clone(), cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        int best = 0; double bestA = 0;
        for (size_t i = 0; i < cnts.size(); ++i) { double a = contourArea(cnts[i]); if (a > bestA) { bestA = a; best = (int)i; } }
        vector<Point> hull; convexHull(cnts[best], hull);
        Mat vis = work[2].clone();
        drawContours(vis, cnts, best, Scalar(255, 0, 0), 2, LINE_AA);                 // bleu : contour
        vector<vector<Point> > hv(1, hull); polylines(vis, hv, true, Scalar(0, 255, 255), 2, LINE_AA);  // jaune : enveloppe
        polygone(vis, det[2].pts, Scalar(0, 255, 0), 3, true);                        // vert : quadrilatere
        putText(vis, "bleu: contour | jaune: enveloppe convexe | vert: quadrilatere", Point(8, 22),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 2, LINE_AA);
        imwrite("fig07_contour_enveloppe_quadrilatere.png", etiqueter(vis, "desk.JPG", 640));
    }

    // Figure 8 : detection automatique pour les 3 images
    {
        vector<Mat> p;
        for (int i = 0; i < 3; ++i) {
            Mat vis = work[i].clone();
            polygone(vis, det[i].pts, Scalar(0, 255, 0), 3, true);
            ostringstream os; os << noms[i] << " : " << (det[i].texture ? "chaine texturee" : "chaine simple")
                                 << " (" << cvRound(det[i].scoreS) << " / " << cvRound(det[i].scoreT) << ")";
            p.push_back(etiqueter(vis, os.str()));
        }
        Mat f; hconcat(p, f);
        imwrite("fig08_detection_automatique.png", f);
    }

    // Figures 10 et 11 : resultats redresses et rendu scanner
    {
        vector<Mat> r10, r11;
        for (int i = 0; i < 3; ++i) {
            Mat res = redresser(orig[i], det[i].pts, ech[i]);
            Mat g, sc; cvtColor(res, g, COLOR_BGR2GRAY);
            adaptiveThreshold(g, sc, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 31, 15);
            r10.push_back(etiqueter(orig[i], string(noms[i]) + " - originale", 380));
            r10.push_back(etiqueter(res, string(noms[i]) + " - redressee", 380));
            r11.push_back(etiqueter(sc, string(noms[i]) + " - rendu scanner", 520));
        }
        Mat f10, f11; hconcat(r10, f10); hconcat(r11, f11);
        imwrite("fig10_resultats_redressement.png", f10);
        imwrite("fig11_rendu_scanner.png", f11);
    }
    cout << "Figures enregistrees dans le dossier d'execution (fig01 a fig11, sauf fig09 : capture d'ecran)." << endl;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Programme principal                                               */
/* ------------------------------------------------------------------ */
int main(int argc, char** argv)
{
    if (argc > 1 && string(argv[1]) == "figures")
        return genererFigures(argc - 2, argv + 2);

    string chemin, mode = "auto";
    if (argc > 1) {
        chemin = argv[1];
        if (argc > 2) mode = argv[2];
    } else {
        cout << "Chemin de l'image : ";
        getline(cin, chemin);
        cout << "Mode (auto / simple / texture) [auto] : ";
        getline(cin, mode);
        if (mode.empty()) mode = "auto";
    }

    chemin = nettoyerChemin(chemin);
    mode = nettoyerChemin(mode);
    if (mode.empty()) mode = "auto";

    // 1) Lecture (imread applique l'orientation EXIF des JPG)
    Mat orig = imread(chemin, IMREAD_COLOR);
    if (orig.empty()) { cerr << "Impossible de lire l'image : [" << chemin << "]\n"
             << "Verifiez le chemin, l'extension (.jpg / .JPG) et que le fichier existe." << endl;
        return -1; }
    cout << "Image chargee : " << chemin << " (" << orig.cols << "x" << orig.rows << ")" << endl;

    // Image de travail reduite (rapidite + noyaux morphologiques coherents)
    double echelle = min(1.0, 900.0 / max(orig.cols, orig.rows));
    Mat work;
    resize(orig, work, Size(), echelle, echelle, INTER_AREA);

    // 2) Detection automatique : evaluation des deux chaines
    Detection det = detecter(work, mode);
    cout << "Soutien des contours - fond simple : " << det.scoreS
         << " | fond texture : " << det.scoreT << endl;
    cout << "Methode retenue : " << (det.texture ? "FOND TEXTURE" : "FOND SIMPLE")
         << (mode == "auto" ? " (choix automatique)" : " (mode force)") << endl;
    vector<Point2f> detectes = det.pts;

    // 3) Affichage du resultat de la detection
    Mat vis = work.clone();
    for (int i = 0; i < 4; ++i) {
        line(vis, detectes[i], detectes[(i + 1) % 4], Scalar(0, 255, 0), 2, LINE_AA);
        circle(vis, detectes[i], 9, Scalar(0, 0, 255), -1, LINE_AA);
    }
    imshow("Source - " + chemin, pourAffichage(work));
    if (!det.masqueS.empty()) imshow("Masque - chaine fond simple", pourAffichage(det.masqueS));
    if (!det.masqueT.empty()) imshow("Masque - chaine fond texture", pourAffichage(det.masqueT));
    imshow("Detection automatique", pourAffichage(vis));
    cout << "Cliquez sur une fenetre d'image puis appuyez sur une touche pour passer a l'ajustement..." << endl;
    waitKey(0);

    // 4) Ajustement interactif
    Etat etat;
    etat.base = work;
    etat.pts = detectes;
    etat.fenetre = "Ajustement des coins";
    etat.f = min(1.0, min(620.0 / work.rows, 1000.0 / work.cols));   // tient dans un ecran de portable
    namedWindow(etat.fenetre, WINDOW_AUTOSIZE);
    setMouseCallback(etat.fenetre, onMouse, &etat);
    dessiner(etat);

    while (true) {
        int key = waitKey(20);
        if (key == 27) { destroyAllWindows(); return 0; }              // ECHAP
        if (key == 13 || key == 'v' || key == 'V') break;              // ENTREE / v
        if (key == 'r' || key == 'R') { etat.pts = detectes; dessiner(etat); }
        if (key == '+' || key == '=') { etat.f = min(2.0, etat.f * 1.15); dessiner(etat); }
        if (key == '-' || key == '_') { etat.f = max(0.2, etat.f / 1.15); dessiner(etat); }
    }

    // 5) Transformation de perspective sur l'image ORIGINALE
    Mat resultat = redresser(orig, etat.pts, echelle);

    // Bonus : rendu "scanner" (binarisation adaptative)
    Mat g2, scan;
    cvtColor(resultat, g2, COLOR_BGR2GRAY);
    adaptiveThreshold(g2, scan, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 31, 15);

    imwrite("resultat_redresse.png", resultat);
    imwrite("resultat_scan.png", scan);
    cout << "Taille de sortie : " << resultat.cols << " x " << resultat.rows << endl;

    destroyWindow(etat.fenetre);
    imshow("Resultat redresse", pourAffichage(resultat));
    imshow("Rendu scanner", pourAffichage(scan));
    waitKey(0);
    destroyAllWindows();
    return 0;
}

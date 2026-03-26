#ifndef COMPUTE_LAPLACIAN_HEADER_FILE
#define COMPUTE_LAPLACIAN_HEADER_FILE

#include <Eigen/Dense>

// computes cot of the angle at vertex pj in the triangle (pi, pj, pk)
double cotangent(const Eigen::Vector3d& pi, const Eigen::Vector3d& pj, const Eigen::Vector3d& pk) {
    Eigen::Vector3d u = pi - pj;
    Eigen::Vector3d v = pk - pj;
    double cosTheta = u.dot(v) / (u.norm() * v.norm());
    double sinTheta = (u.cross(v)).norm() / (u.norm() * v.norm());
    return cosTheta / sinTheta;
}

void compute_laplacian(const Eigen::MatrixXd& V,
                       const Eigen::MatrixXi& F,
                       const Eigen::MatrixXi& E,
                       const Eigen::MatrixXi& EF,
                       const Eigen::VectorXi& boundEMask,
                       Eigen::SparseMatrix<double>& d0,
                       Eigen::SparseMatrix<double>& W,
                       Eigen::VectorXd& vorAreas){
    
    using namespace Eigen;
    using namespace std;
    d0.resize(E.rows(), V.rows());
    W.resize(E.rows(), E.rows());
    vorAreas = VectorXd::Ones(V.rows());
   
    // d0(i, source(ei)) = -1, d0(i, target(ei)) = 1
    vector<Triplet<double>> d0Triplets;
    d0Triplets.reserve(2 * E.rows());
    for (int i = 0; i < E.rows(); ++i) {
        int source = E(i, 0);
        int target = E(i, 1);
        d0Triplets.emplace_back(i, source, -1.0);
        d0Triplets.emplace_back(i, target, 1.0);
    }
    d0.setFromTriplets(d0Triplets.begin(), d0Triplets.end());

    // Per-face areas for Voronoi area computation
    VectorXd faceAreas(F.rows());
    for (int f = 0; f < F.rows(); f++) {
        Vector3d v0 = V.row(F(f, 0));
        Vector3d v1 = V.row(F(f, 1));
        Vector3d v2 = V.row(F(f, 2));
        faceAreas(f) = 0.5 * ((v1 - v0).cross(v2 - v0)).norm();
    }

    // Voronoi areas: A(v) = 1/3 * sum of adjacent face areas
    for (int f = 0; f < F.rows(); f++) {
        vorAreas(F(f, 0)) += faceAreas(f) / 3.0;
        vorAreas(F(f, 1)) += faceAreas(f) / 3.0;
        vorAreas(F(f, 2)) += faceAreas(f) / 3.0;
    }

    // At boundary W(i, i) = 0.5 * cot(alpha_j)
    // At interior W(i, i) = 0.5 * (cot(alpha_j) + cot(alpha_l))
    
    vector<Triplet<double>> WTriplets;
    WTriplets.reserve(E.rows());
    for (int e = 0; e < E.rows(); ++e) {
        int source = E(e, 0); //i
        int target = E(e, 1); //k

        // left face
        int fL = EF(e, 0);
        int jLocal = EF(e, 1);
        int j = F(fL, jLocalL); //j

        double cot_j = cotangent(V.row(source), V.row(j), V.row(target));
        double weight = 0.5 * cot_j;

        // right face if not boundary edge
        if (boundEMask(e) == 0) {
            int fR = EF(e, 2);
            int lLocal = EF(e, 3);
            int l = F(fR, lLocal); //l

            double cot_l = cotangent(V.row(source), V.row(l), V.row(target));
            weight += 0.5 * cot_l;
        }

        WTriplets.emplace_back(e, e, weight);
    }
    W.setFromTriplets(WTriplets.begin(), WTriplets.end());
}



#endif

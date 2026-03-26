#ifndef COMPUTE_ANGLE_DEFECT_HEADER_FILE
#define COMPUTE_ANGLE_DEFECT_HEADER_FILE

#include <Eigen/Dense>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Eigen::VectorXd compute_angle_defect(const Eigen::MatrixXd& V,
                                     const Eigen::MatrixXi& F,
                                     const Eigen::VectorXi& boundVMask){
    
    //Stub
    Eigen::VectorXd G = Eigen::VectorXd::Zero(V.rows());

    //each face, compute angle at each vertex, add to G
    for (int f = 0; f < F.rows(); ++f) {
        //face vertices
        int v0 = F(f, 0);
        int v1 = F(f, 1);
        int v2 = F(f, 2);

        //face edges normalized
        Eigen::Vector3d e0 = (V.row(v1) - V.row(v0)).normalized();
        Eigen::Vector3d e1 = (V.row(v2) - V.row(v1)).normalized();
        Eigen::Vector3d e2 = (V.row(v0) - V.row(v2)).normalized();

        //angles at each vertex
        double a0_2 = std::acos(e0.dot(-e2));
        double a1_0 = std::acos(e1.dot(-e0));
        double a2_1 = std::acos(e2.dot(-e1));

        G(v0) += a0_2;
        G(v1) += a1_0;
        G(v2) += a2_1;
    }

    //subtract from pi (boundary) or 2pi (interior)
    for (int v = 0; v < V.rows(); v++) {
        if (boundVMask(v) == 1) {
            G(v) = M_PI - G(v);
        } else {
            G(v) = 2 * M_PI - G(v);
        }
    }

    return G;
}


#endif

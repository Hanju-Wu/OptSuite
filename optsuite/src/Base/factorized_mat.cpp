//
// Created by liuhy on 10/14/21.
//
#include "OptSuite/Base/factorized_mat.h"

namespace OptSuite {
    namespace Base {
        template<typename dtype>
        dtype FactorizedMat<dtype>::dot(const Variable<dtype> &other) const {
            const auto *other_ptr_f = dynamic_cast<const FactorizedMat *>(&other);
            if (other_ptr_f)
                return this->dot(*other_ptr_f);
            else
                return Variable<dtype>::dot(other);
        }

        template<typename dtype>
        dtype FactorizedMat<dtype>::dot(const FactorizedMat& other) const {
            // quick return if possible
            if (rank() == 0)
                return 0_s;
            // <U1'V1, U2'V2> = <V1V2', U1U2'>
            tmpV = V_ * other.V_.adjoint();
            return tmpV.conjugate().cwiseProduct(U_ * other.U_.adjoint()).sum();
        }

        template<typename dtype>
        dtype FactorizedMat<dtype>::dot(const Ref<const spmat_t>& other) const {
            // quick return if possible
            if (rank() == 0)
                return 0_s;

            // <U'V, S> = <V, US>
            tmpV = U_ * other;
            return V_.conjugate().cwiseProduct(tmpV).sum();
        }

        template<typename dtype>
        dtype FactorizedMat<dtype>::dot(const Ref<const mat_t>& other) const {
            // quick return if possible
            if (rank() == 0)
                return 0_s;

            // <U'V, S> = <V, US>
            tmpV = U_ * other;
            return V_.conjugate().cwiseProduct(tmpV).sum();
        }

        template class FactorizedMat<Scalar>;
        template class FactorizedMat<ComplexScalar>;
    }
}

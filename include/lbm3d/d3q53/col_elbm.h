#pragma once

#include "common.h"
#include "eq.h"


template <typename TRAITS, typename LBM_EQ = D3Q53_EQ<TRAITS>>
struct D3Q53_ELBM : D3Q53_COMMON<TRAITS, LBM_EQ>
{
	using dreal = typename TRAITS::dreal;

    static constexpr const dreal precision = (dreal)(0.0000001);
    static constexpr const int escape = 10;

	static constexpr const char* id = "ELBM_D3Q53";

    template <typename LBM_KS>
	__cuda_callable__ static void collision(LBM_KS &KS, bool equilibrium = false){
        const dreal beta1 = 1./(no2*KS.lbmViscosity/KS.T0 + no1);
        const dreal tau =  (no2*KS.lbmViscosity/KS.T0 + no1);

        dreal feq[LBM_KS::Q];
		#ifdef USE_FORCING
		// for no force equilibrium
		KS.vx -= 1.0*KS.fx/2;
        KS.vy -= 1.0*KS.fy/2;
		#endif

        dreal A = KS.A;
        dreal B1 = KS.B1;
        dreal B2 = KS.B2;
        dreal B3 = KS.B3;
        findEntropicEquilibrium(A, B1, B2,B3, KS);
        calculateEquilibriumDistribution(feq,A,B1,B2,B3,KS);

        // Update the Lagrange multipliers to converge faster on the next iteration
        KS.A = A;
        KS.B1 = B1;
        KS.B2 = B2;
        KS.B3 = B3;
        if(equilibrium){
            for(int id = 0; id < LBM_KS::Q; id++){
                KS.f[id] = feq[id];
            }
            return;
        }
		#ifdef USE_FORCING
		// Full force equilibrium
		dreal feqF[LBM_KS::Q];
        for(int i = 0; i < LBM_KS::Q; i++){
            feqF[i] = feq[i];
        }
        KS.vx += 1.0*KS.fx;
        KS.vy += 1.0*KS.fy;
		findEntropicEquilibrium(A, B1, B2,B3, KS);
        calculateEquilibriumDistribution(feqF,A,B1,B2,B3,KS);


        KS.vx -= 1.0*KS.fx/2;
        KS.vy -= 1.0*KS.fy/2;
		#endif

		for(int id = 0; id < LBM_KS::Q; id++){
            KS.f[id] += 2.*beta1*(feq[id] - KS.f[id])
			#ifdef USE_FORCING
			+ (feqF[id] - feq[id])
			#endif
			;
        }
	}

    template <typename LBM_KS>
    __cuda_callable__ static void findEntropicEquilibrium(dreal &A, dreal &B1, dreal &B2, dreal &B3, LBM_KS &KS){
        // Newton method variables declaration
        int iteration = 1;
        // Declaration of the function that is being minimized (F) and its norm (norm)
        dreal F[4];
        calculateConservationLawsRightHandSide(F,A,B1,B2,B3,KS);
        dreal norm = F[0]*F[0]+F[1]*F[1]+F[2]*F[2]+F[3]*F[3];
        // Main Newton method loop
        while(norm> precision*precision && iteration < escape){
            dreal oldNorm = norm;
            performNewtonMethodStep<LBM_KS>(F,A,B1,B2,B3);
            // Calculate the norm and go to the next iteration
            calculateConservationLawsRightHandSide(F,A,B1,B2,B3,KS);
            norm = F[0]*F[0]+F[1]*F[1]+F[2]*F[2]+F[3]*F[3];
            int insideIter = 1;
            while(norm > oldNorm && insideIter < escape){
                A = (A+KS.A)/2.;
                B1 = (B1+KS.B1)/2.;
                B2 = (B2+KS.B2)/2.;
                B3 = (B3+KS.B3)/2.;
                calculateConservationLawsRightHandSide(F,A,B1,B2,B3,KS);
                norm = F[0]*F[0]+F[1]*F[1]+F[2]*F[2]+F[3]*F[3];
                insideIter++;
            }
            iteration++;
        }
    }

    template <typename LBM_KS>
    __cuda_callable__ static void performNewtonMethodStep(dreal (&F)[4], dreal &A, dreal &B1, dreal &B2, dreal &B3){
        // [A B1 B2 B3] = [A B1 B2 B3] - inverseJ * F
        // An optimalized function to symbolically compute the inverseJ matrix of the function F
        // inverseJ is then used in Newton's method
        dreal J[4][4];
        {
        J[0][0] = -(B1*B1*B1)*LBM_KS::id_to_weight(52) - (B1*B1)*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(51) - (B1*B1)*(B2*B2)*LBM_KS::id_to_weight(50) - (B1*B1)*(B2*B2)*LBM_KS::id_to_weight(49)/(B3*B3) - (B1*B1)*(B3*B3)*LBM_KS::id_to_weight(48) - (B1*B1)*LBM_KS::id_to_weight(47)/(B3*B3) - (B1*B1)*(B3*B3)*LBM_KS::id_to_weight(46)/(B2*B2) - (B1*B1)*LBM_KS::id_to_weight(45)/(B2*B2) - (B1*B1)*LBM_KS::id_to_weight(44)/((B2*B2)*(B3*B3)) - B1*B2*B3*LBM_KS::id_to_weight(43) - B1*B2*LBM_KS::id_to_weight(42) - B1*B2*LBM_KS::id_to_weight(41)/B3 - B1*B3*LBM_KS::id_to_weight(40) - B1*LBM_KS::id_to_weight(39) - B1*LBM_KS::id_to_weight(38)/B3 - B1*B3*LBM_KS::id_to_weight(37)/B2 - B1*LBM_KS::id_to_weight(36)/B2 - B1*LBM_KS::id_to_weight(35)/(B2*B3) - (B2*B2*B2)*LBM_KS::id_to_weight(34) - (B2*B2)*(B3*B3)*LBM_KS::id_to_weight(33) - (B2*B2)*LBM_KS::id_to_weight(32)/(B3*B3) - B2*B3*LBM_KS::id_to_weight(31) - B2*LBM_KS::id_to_weight(30) - B2*LBM_KS::id_to_weight(29)/B3 - (B3*B3*B3)*LBM_KS::id_to_weight(28) - B3*LBM_KS::id_to_weight(27) - LBM_KS::id_to_weight(26) - LBM_KS::id_to_weight(25)/B3 - LBM_KS::id_to_weight(24)/(B3*B3*B3) - B3*LBM_KS::id_to_weight(23)/B2 - LBM_KS::id_to_weight(22)/B2 - LBM_KS::id_to_weight(21)/(B2*B3) - (B3*B3)*LBM_KS::id_to_weight(20)/(B2*B2) - LBM_KS::id_to_weight(19)/((B2*B2)*(B3*B3)) - LBM_KS::id_to_weight(18)/(B2*B2*B2) - B2*B3*LBM_KS::id_to_weight(17)/B1 - B2*LBM_KS::id_to_weight(16)/B1 - B2*LBM_KS::id_to_weight(15)/(B1*B3) - B3*LBM_KS::id_to_weight(14)/B1 - LBM_KS::id_to_weight(13)/B1 - LBM_KS::id_to_weight(12)/(B1*B3) - B3*LBM_KS::id_to_weight(11)/(B1*B2) - LBM_KS::id_to_weight(10)/(B1*B2) - LBM_KS::id_to_weight(9)/(B1*B2*B3) - (B2*B2)*(B3*B3)*LBM_KS::id_to_weight(8)/(B1*B1) - (B2*B2)*LBM_KS::id_to_weight(7)/(B1*B1) - (B2*B2)*LBM_KS::id_to_weight(6)/((B1*B1)*(B3*B3)) - (B3*B3)*LBM_KS::id_to_weight(5)/(B1*B1) - LBM_KS::id_to_weight(4)/((B1*B1)*(B3*B3)) - (B3*B3)*LBM_KS::id_to_weight(3)/((B1*B1)*(B2*B2)) - LBM_KS::id_to_weight(2)/((B1*B1)*(B2*B2)) - LBM_KS::id_to_weight(1)/((B1*B1)*(B2*B2)*(B3*B3)) - LBM_KS::id_to_weight(0)/(B1*B1*B1);
J[0][1] = -3*A*(B1*B1)*LBM_KS::id_to_weight(52) - 2*A*B1*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(51) - 2*A*B1*(B2*B2)*LBM_KS::id_to_weight(50) - 2*A*B1*(B2*B2)*LBM_KS::id_to_weight(49)/(B3*B3) - 2*A*B1*(B3*B3)*LBM_KS::id_to_weight(48) - 2*A*B1*LBM_KS::id_to_weight(47)/(B3*B3) - 2*A*B1*(B3*B3)*LBM_KS::id_to_weight(46)/(B2*B2) - 2*A*B1*LBM_KS::id_to_weight(45)/(B2*B2) - 2*A*B1*LBM_KS::id_to_weight(44)/((B2*B2)*(B3*B3)) - A*B2*B3*LBM_KS::id_to_weight(43) - A*B2*LBM_KS::id_to_weight(42) - A*B2*LBM_KS::id_to_weight(41)/B3 - A*B3*LBM_KS::id_to_weight(40) - A*LBM_KS::id_to_weight(39) - A*LBM_KS::id_to_weight(38)/B3 - A*B3*LBM_KS::id_to_weight(37)/B2 - A*LBM_KS::id_to_weight(36)/B2 - A*LBM_KS::id_to_weight(35)/(B2*B3) + A*B2*B3*LBM_KS::id_to_weight(17)/(B1*B1) + A*B2*LBM_KS::id_to_weight(16)/(B1*B1) + A*B2*LBM_KS::id_to_weight(15)/((B1*B1)*B3) + A*B3*LBM_KS::id_to_weight(14)/(B1*B1) + A*LBM_KS::id_to_weight(13)/(B1*B1) + A*LBM_KS::id_to_weight(12)/((B1*B1)*B3) + A*B3*LBM_KS::id_to_weight(11)/((B1*B1)*B2) + A*LBM_KS::id_to_weight(10)/((B1*B1)*B2) + A*LBM_KS::id_to_weight(9)/((B1*B1)*B2*B3) + 2*A*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(8)/(B1*B1*B1) + 2*A*(B2*B2)*LBM_KS::id_to_weight(7)/(B1*B1*B1) + 2*A*(B2*B2)*LBM_KS::id_to_weight(6)/((B1*B1*B1)*(B3*B3)) + 2*A*(B3*B3)*LBM_KS::id_to_weight(5)/(B1*B1*B1) + 2*A*LBM_KS::id_to_weight(4)/((B1*B1*B1)*(B3*B3)) + 2*A*(B3*B3)*LBM_KS::id_to_weight(3)/((B1*B1*B1)*(B2*B2)) + 2*A*LBM_KS::id_to_weight(2)/((B1*B1*B1)*(B2*B2)) + 2*A*LBM_KS::id_to_weight(1)/((B1*B1*B1)*(B2*B2)*(B3*B3)) + 3*A*LBM_KS::id_to_weight(0)/(B1*B1*B1*B1);
J[0][2] = -2*A*(B1*B1)*B2*(B3*B3)*LBM_KS::id_to_weight(51) - 2*A*(B1*B1)*B2*LBM_KS::id_to_weight(50) - 2*A*(B1*B1)*B2*LBM_KS::id_to_weight(49)/(B3*B3) + 2*A*(B1*B1)*(B3*B3)*LBM_KS::id_to_weight(46)/(B2*B2*B2) + 2*A*(B1*B1)*LBM_KS::id_to_weight(45)/(B2*B2*B2) + 2*A*(B1*B1)*LBM_KS::id_to_weight(44)/((B2*B2*B2)*(B3*B3)) - A*B1*B3*LBM_KS::id_to_weight(43) - A*B1*LBM_KS::id_to_weight(42) - A*B1*LBM_KS::id_to_weight(41)/B3 + A*B1*B3*LBM_KS::id_to_weight(37)/(B2*B2) + A*B1*LBM_KS::id_to_weight(36)/(B2*B2) + A*B1*LBM_KS::id_to_weight(35)/((B2*B2)*B3) - 3*A*(B2*B2)*LBM_KS::id_to_weight(34) - 2*A*B2*(B3*B3)*LBM_KS::id_to_weight(33) - 2*A*B2*LBM_KS::id_to_weight(32)/(B3*B3) - A*B3*LBM_KS::id_to_weight(31) - A*LBM_KS::id_to_weight(30) - A*LBM_KS::id_to_weight(29)/B3 + A*B3*LBM_KS::id_to_weight(23)/(B2*B2) + A*LBM_KS::id_to_weight(22)/(B2*B2) + A*LBM_KS::id_to_weight(21)/((B2*B2)*B3) + 2*A*(B3*B3)*LBM_KS::id_to_weight(20)/(B2*B2*B2) + 2*A*LBM_KS::id_to_weight(19)/((B2*B2*B2)*(B3*B3)) + 3*A*LBM_KS::id_to_weight(18)/(B2*B2*B2*B2) - A*B3*LBM_KS::id_to_weight(17)/B1 - A*LBM_KS::id_to_weight(16)/B1 - A*LBM_KS::id_to_weight(15)/(B1*B3) + A*B3*LBM_KS::id_to_weight(11)/(B1*(B2*B2)) + A*LBM_KS::id_to_weight(10)/(B1*(B2*B2)) + A*LBM_KS::id_to_weight(9)/(B1*(B2*B2)*B3) - 2*A*B2*(B3*B3)*LBM_KS::id_to_weight(8)/(B1*B1) - 2*A*B2*LBM_KS::id_to_weight(7)/(B1*B1) - 2*A*B2*LBM_KS::id_to_weight(6)/((B1*B1)*(B3*B3)) + 2*A*(B3*B3)*LBM_KS::id_to_weight(3)/((B1*B1)*(B2*B2*B2)) + 2*A*LBM_KS::id_to_weight(2)/((B1*B1)*(B2*B2*B2)) + 2*A*LBM_KS::id_to_weight(1)/((B1*B1)*(B2*B2*B2)*(B3*B3));
J[0][3] = -2*A*(B1*B1)*(B2*B2)*B3*LBM_KS::id_to_weight(51) + 2*A*(B1*B1)*(B2*B2)*LBM_KS::id_to_weight(49)/(B3*B3*B3) - 2*A*(B1*B1)*B3*LBM_KS::id_to_weight(48) + 2*A*(B1*B1)*LBM_KS::id_to_weight(47)/(B3*B3*B3) - 2*A*(B1*B1)*B3*LBM_KS::id_to_weight(46)/(B2*B2) + 2*A*(B1*B1)*LBM_KS::id_to_weight(44)/((B2*B2)*(B3*B3*B3)) - A*B1*B2*LBM_KS::id_to_weight(43) + A*B1*B2*LBM_KS::id_to_weight(41)/(B3*B3) - A*B1*LBM_KS::id_to_weight(40) + A*B1*LBM_KS::id_to_weight(38)/(B3*B3) - A*B1*LBM_KS::id_to_weight(37)/B2 + A*B1*LBM_KS::id_to_weight(35)/(B2*(B3*B3)) - 2*A*(B2*B2)*B3*LBM_KS::id_to_weight(33) + 2*A*(B2*B2)*LBM_KS::id_to_weight(32)/(B3*B3*B3) - A*B2*LBM_KS::id_to_weight(31) + A*B2*LBM_KS::id_to_weight(29)/(B3*B3) - 3*A*(B3*B3)*LBM_KS::id_to_weight(28) - A*LBM_KS::id_to_weight(27) + A*LBM_KS::id_to_weight(25)/(B3*B3) + 3*A*LBM_KS::id_to_weight(24)/(B3*B3*B3*B3) - A*LBM_KS::id_to_weight(23)/B2 + A*LBM_KS::id_to_weight(21)/(B2*(B3*B3)) - 2*A*B3*LBM_KS::id_to_weight(20)/(B2*B2) + 2*A*LBM_KS::id_to_weight(19)/((B2*B2)*(B3*B3*B3)) - A*B2*LBM_KS::id_to_weight(17)/B1 + A*B2*LBM_KS::id_to_weight(15)/(B1*(B3*B3)) - A*LBM_KS::id_to_weight(14)/B1 + A*LBM_KS::id_to_weight(12)/(B1*(B3*B3)) - A*LBM_KS::id_to_weight(11)/(B1*B2) + A*LBM_KS::id_to_weight(9)/(B1*B2*(B3*B3)) - 2*A*(B2*B2)*B3*LBM_KS::id_to_weight(8)/(B1*B1) + 2*A*(B2*B2)*LBM_KS::id_to_weight(6)/((B1*B1)*(B3*B3*B3)) - 2*A*B3*LBM_KS::id_to_weight(5)/(B1*B1) + 2*A*LBM_KS::id_to_weight(4)/((B1*B1)*(B3*B3*B3)) - 2*A*B3*LBM_KS::id_to_weight(3)/((B1*B1)*(B2*B2)) + 2*A*LBM_KS::id_to_weight(1)/((B1*B1)*(B2*B2)*(B3*B3*B3));
J[1][0] = -3*(B1*B1*B1)*LBM_KS::id_to_weight(52) - 2*(B1*B1)*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(51) - 2*(B1*B1)*(B2*B2)*LBM_KS::id_to_weight(50) - 2*(B1*B1)*(B2*B2)*LBM_KS::id_to_weight(49)/(B3*B3) - 2*(B1*B1)*(B3*B3)*LBM_KS::id_to_weight(48) - 2*(B1*B1)*LBM_KS::id_to_weight(47)/(B3*B3) - 2*(B1*B1)*(B3*B3)*LBM_KS::id_to_weight(46)/(B2*B2) - 2*(B1*B1)*LBM_KS::id_to_weight(45)/(B2*B2) - 2*(B1*B1)*LBM_KS::id_to_weight(44)/((B2*B2)*(B3*B3)) - B1*B2*B3*LBM_KS::id_to_weight(43) - B1*B2*LBM_KS::id_to_weight(42) - B1*B2*LBM_KS::id_to_weight(41)/B3 - B1*B3*LBM_KS::id_to_weight(40) - B1*LBM_KS::id_to_weight(39) - B1*LBM_KS::id_to_weight(38)/B3 - B1*B3*LBM_KS::id_to_weight(37)/B2 - B1*LBM_KS::id_to_weight(36)/B2 - B1*LBM_KS::id_to_weight(35)/(B2*B3) + B2*B3*LBM_KS::id_to_weight(17)/B1 + B2*LBM_KS::id_to_weight(16)/B1 + B2*LBM_KS::id_to_weight(15)/(B1*B3) + B3*LBM_KS::id_to_weight(14)/B1 + LBM_KS::id_to_weight(13)/B1 + LBM_KS::id_to_weight(12)/(B1*B3) + B3*LBM_KS::id_to_weight(11)/(B1*B2) + LBM_KS::id_to_weight(10)/(B1*B2) + LBM_KS::id_to_weight(9)/(B1*B2*B3) + 2*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(8)/(B1*B1) + 2*(B2*B2)*LBM_KS::id_to_weight(7)/(B1*B1) + 2*(B2*B2)*LBM_KS::id_to_weight(6)/((B1*B1)*(B3*B3)) + 2*(B3*B3)*LBM_KS::id_to_weight(5)/(B1*B1) + 2*LBM_KS::id_to_weight(4)/((B1*B1)*(B3*B3)) + 2*(B3*B3)*LBM_KS::id_to_weight(3)/((B1*B1)*(B2*B2)) + 2*LBM_KS::id_to_weight(2)/((B1*B1)*(B2*B2)) + 2*LBM_KS::id_to_weight(1)/((B1*B1)*(B2*B2)*(B3*B3)) + 3*LBM_KS::id_to_weight(0)/(B1*B1*B1);
J[1][1] = -9*A*(B1*B1)*LBM_KS::id_to_weight(52) - 4*A*B1*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(51) - 4*A*B1*(B2*B2)*LBM_KS::id_to_weight(50) - 4*A*B1*(B2*B2)*LBM_KS::id_to_weight(49)/(B3*B3) - 4*A*B1*(B3*B3)*LBM_KS::id_to_weight(48) - 4*A*B1*LBM_KS::id_to_weight(47)/(B3*B3) - 4*A*B1*(B3*B3)*LBM_KS::id_to_weight(46)/(B2*B2) - 4*A*B1*LBM_KS::id_to_weight(45)/(B2*B2) - 4*A*B1*LBM_KS::id_to_weight(44)/((B2*B2)*(B3*B3)) - A*B2*B3*LBM_KS::id_to_weight(43) - A*B2*LBM_KS::id_to_weight(42) - A*B2*LBM_KS::id_to_weight(41)/B3 - A*B3*LBM_KS::id_to_weight(40) - A*LBM_KS::id_to_weight(39) - A*LBM_KS::id_to_weight(38)/B3 - A*B3*LBM_KS::id_to_weight(37)/B2 - A*LBM_KS::id_to_weight(36)/B2 - A*LBM_KS::id_to_weight(35)/(B2*B3) - A*B2*B3*LBM_KS::id_to_weight(17)/(B1*B1) - A*B2*LBM_KS::id_to_weight(16)/(B1*B1) - A*B2*LBM_KS::id_to_weight(15)/((B1*B1)*B3) - A*B3*LBM_KS::id_to_weight(14)/(B1*B1) - A*LBM_KS::id_to_weight(13)/(B1*B1) - A*LBM_KS::id_to_weight(12)/((B1*B1)*B3) - A*B3*LBM_KS::id_to_weight(11)/((B1*B1)*B2) - A*LBM_KS::id_to_weight(10)/((B1*B1)*B2) - A*LBM_KS::id_to_weight(9)/((B1*B1)*B2*B3) - 4*A*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(8)/(B1*B1*B1) - 4*A*(B2*B2)*LBM_KS::id_to_weight(7)/(B1*B1*B1) - 4*A*(B2*B2)*LBM_KS::id_to_weight(6)/((B1*B1*B1)*(B3*B3)) - 4*A*(B3*B3)*LBM_KS::id_to_weight(5)/(B1*B1*B1) - 4*A*LBM_KS::id_to_weight(4)/((B1*B1*B1)*(B3*B3)) - 4*A*(B3*B3)*LBM_KS::id_to_weight(3)/((B1*B1*B1)*(B2*B2)) - 4*A*LBM_KS::id_to_weight(2)/((B1*B1*B1)*(B2*B2)) - 4*A*LBM_KS::id_to_weight(1)/((B1*B1*B1)*(B2*B2)*(B3*B3)) - 9*A*LBM_KS::id_to_weight(0)/(B1*B1*B1*B1);
J[1][2] = -4*A*(B1*B1)*B2*(B3*B3)*LBM_KS::id_to_weight(51) - 4*A*(B1*B1)*B2*LBM_KS::id_to_weight(50) - 4*A*(B1*B1)*B2*LBM_KS::id_to_weight(49)/(B3*B3) + 4*A*(B1*B1)*(B3*B3)*LBM_KS::id_to_weight(46)/(B2*B2*B2) + 4*A*(B1*B1)*LBM_KS::id_to_weight(45)/(B2*B2*B2) + 4*A*(B1*B1)*LBM_KS::id_to_weight(44)/((B2*B2*B2)*(B3*B3)) - A*B1*B3*LBM_KS::id_to_weight(43) - A*B1*LBM_KS::id_to_weight(42) - A*B1*LBM_KS::id_to_weight(41)/B3 + A*B1*B3*LBM_KS::id_to_weight(37)/(B2*B2) + A*B1*LBM_KS::id_to_weight(36)/(B2*B2) + A*B1*LBM_KS::id_to_weight(35)/((B2*B2)*B3) + A*B3*LBM_KS::id_to_weight(17)/B1 + A*LBM_KS::id_to_weight(16)/B1 + A*LBM_KS::id_to_weight(15)/(B1*B3) - A*B3*LBM_KS::id_to_weight(11)/(B1*(B2*B2)) - A*LBM_KS::id_to_weight(10)/(B1*(B2*B2)) - A*LBM_KS::id_to_weight(9)/(B1*(B2*B2)*B3) + 4*A*B2*(B3*B3)*LBM_KS::id_to_weight(8)/(B1*B1) + 4*A*B2*LBM_KS::id_to_weight(7)/(B1*B1) + 4*A*B2*LBM_KS::id_to_weight(6)/((B1*B1)*(B3*B3)) - 4*A*(B3*B3)*LBM_KS::id_to_weight(3)/((B1*B1)*(B2*B2*B2)) - 4*A*LBM_KS::id_to_weight(2)/((B1*B1)*(B2*B2*B2)) - 4*A*LBM_KS::id_to_weight(1)/((B1*B1)*(B2*B2*B2)*(B3*B3));
J[1][3] = -4*A*(B1*B1)*(B2*B2)*B3*LBM_KS::id_to_weight(51) + 4*A*(B1*B1)*(B2*B2)*LBM_KS::id_to_weight(49)/(B3*B3*B3) - 4*A*(B1*B1)*B3*LBM_KS::id_to_weight(48) + 4*A*(B1*B1)*LBM_KS::id_to_weight(47)/(B3*B3*B3) - 4*A*(B1*B1)*B3*LBM_KS::id_to_weight(46)/(B2*B2) + 4*A*(B1*B1)*LBM_KS::id_to_weight(44)/((B2*B2)*(B3*B3*B3)) - A*B1*B2*LBM_KS::id_to_weight(43) + A*B1*B2*LBM_KS::id_to_weight(41)/(B3*B3) - A*B1*LBM_KS::id_to_weight(40) + A*B1*LBM_KS::id_to_weight(38)/(B3*B3) - A*B1*LBM_KS::id_to_weight(37)/B2 + A*B1*LBM_KS::id_to_weight(35)/(B2*(B3*B3)) + A*B2*LBM_KS::id_to_weight(17)/B1 - A*B2*LBM_KS::id_to_weight(15)/(B1*(B3*B3)) + A*LBM_KS::id_to_weight(14)/B1 - A*LBM_KS::id_to_weight(12)/(B1*(B3*B3)) + A*LBM_KS::id_to_weight(11)/(B1*B2) - A*LBM_KS::id_to_weight(9)/(B1*B2*(B3*B3)) + 4*A*(B2*B2)*B3*LBM_KS::id_to_weight(8)/(B1*B1) - 4*A*(B2*B2)*LBM_KS::id_to_weight(6)/((B1*B1)*(B3*B3*B3)) + 4*A*B3*LBM_KS::id_to_weight(5)/(B1*B1) - 4*A*LBM_KS::id_to_weight(4)/((B1*B1)*(B3*B3*B3)) + 4*A*B3*LBM_KS::id_to_weight(3)/((B1*B1)*(B2*B2)) - 4*A*LBM_KS::id_to_weight(1)/((B1*B1)*(B2*B2)*(B3*B3*B3));
J[2][0] = -2*(B1*B1)*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(51) - 2*(B1*B1)*(B2*B2)*LBM_KS::id_to_weight(50) - 2*(B1*B1)*(B2*B2)*LBM_KS::id_to_weight(49)/(B3*B3) + 2*(B1*B1)*(B3*B3)*LBM_KS::id_to_weight(46)/(B2*B2) + 2*(B1*B1)*LBM_KS::id_to_weight(45)/(B2*B2) + 2*(B1*B1)*LBM_KS::id_to_weight(44)/((B2*B2)*(B3*B3)) - B1*B2*B3*LBM_KS::id_to_weight(43) - B1*B2*LBM_KS::id_to_weight(42) - B1*B2*LBM_KS::id_to_weight(41)/B3 + B1*B3*LBM_KS::id_to_weight(37)/B2 + B1*LBM_KS::id_to_weight(36)/B2 + B1*LBM_KS::id_to_weight(35)/(B2*B3) - 3*(B2*B2*B2)*LBM_KS::id_to_weight(34) - 2*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(33) - 2*(B2*B2)*LBM_KS::id_to_weight(32)/(B3*B3) - B2*B3*LBM_KS::id_to_weight(31) - B2*LBM_KS::id_to_weight(30) - B2*LBM_KS::id_to_weight(29)/B3 + B3*LBM_KS::id_to_weight(23)/B2 + LBM_KS::id_to_weight(22)/B2 + LBM_KS::id_to_weight(21)/(B2*B3) + 2*(B3*B3)*LBM_KS::id_to_weight(20)/(B2*B2) + 2*LBM_KS::id_to_weight(19)/((B2*B2)*(B3*B3)) + 3*LBM_KS::id_to_weight(18)/(B2*B2*B2) - B2*B3*LBM_KS::id_to_weight(17)/B1 - B2*LBM_KS::id_to_weight(16)/B1 - B2*LBM_KS::id_to_weight(15)/(B1*B3) + B3*LBM_KS::id_to_weight(11)/(B1*B2) + LBM_KS::id_to_weight(10)/(B1*B2) + LBM_KS::id_to_weight(9)/(B1*B2*B3) - 2*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(8)/(B1*B1) - 2*(B2*B2)*LBM_KS::id_to_weight(7)/(B1*B1) - 2*(B2*B2)*LBM_KS::id_to_weight(6)/((B1*B1)*(B3*B3)) + 2*(B3*B3)*LBM_KS::id_to_weight(3)/((B1*B1)*(B2*B2)) + 2*LBM_KS::id_to_weight(2)/((B1*B1)*(B2*B2)) + 2*LBM_KS::id_to_weight(1)/((B1*B1)*(B2*B2)*(B3*B3));
J[2][1] = -4*A*B1*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(51) - 4*A*B1*(B2*B2)*LBM_KS::id_to_weight(50) - 4*A*B1*(B2*B2)*LBM_KS::id_to_weight(49)/(B3*B3) + 4*A*B1*(B3*B3)*LBM_KS::id_to_weight(46)/(B2*B2) + 4*A*B1*LBM_KS::id_to_weight(45)/(B2*B2) + 4*A*B1*LBM_KS::id_to_weight(44)/((B2*B2)*(B3*B3)) - A*B2*B3*LBM_KS::id_to_weight(43) - A*B2*LBM_KS::id_to_weight(42) - A*B2*LBM_KS::id_to_weight(41)/B3 + A*B3*LBM_KS::id_to_weight(37)/B2 + A*LBM_KS::id_to_weight(36)/B2 + A*LBM_KS::id_to_weight(35)/(B2*B3) + A*B2*B3*LBM_KS::id_to_weight(17)/(B1*B1) + A*B2*LBM_KS::id_to_weight(16)/(B1*B1) + A*B2*LBM_KS::id_to_weight(15)/((B1*B1)*B3) - A*B3*LBM_KS::id_to_weight(11)/((B1*B1)*B2) - A*LBM_KS::id_to_weight(10)/((B1*B1)*B2) - A*LBM_KS::id_to_weight(9)/((B1*B1)*B2*B3) + 4*A*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(8)/(B1*B1*B1) + 4*A*(B2*B2)*LBM_KS::id_to_weight(7)/(B1*B1*B1) + 4*A*(B2*B2)*LBM_KS::id_to_weight(6)/((B1*B1*B1)*(B3*B3)) - 4*A*(B3*B3)*LBM_KS::id_to_weight(3)/((B1*B1*B1)*(B2*B2)) - 4*A*LBM_KS::id_to_weight(2)/((B1*B1*B1)*(B2*B2)) - 4*A*LBM_KS::id_to_weight(1)/((B1*B1*B1)*(B2*B2)*(B3*B3));
J[2][2] = -4*A*(B1*B1)*B2*(B3*B3)*LBM_KS::id_to_weight(51) - 4*A*(B1*B1)*B2*LBM_KS::id_to_weight(50) - 4*A*(B1*B1)*B2*LBM_KS::id_to_weight(49)/(B3*B3) - 4*A*(B1*B1)*(B3*B3)*LBM_KS::id_to_weight(46)/(B2*B2*B2) - 4*A*(B1*B1)*LBM_KS::id_to_weight(45)/(B2*B2*B2) - 4*A*(B1*B1)*LBM_KS::id_to_weight(44)/((B2*B2*B2)*(B3*B3)) - A*B1*B3*LBM_KS::id_to_weight(43) - A*B1*LBM_KS::id_to_weight(42) - A*B1*LBM_KS::id_to_weight(41)/B3 - A*B1*B3*LBM_KS::id_to_weight(37)/(B2*B2) - A*B1*LBM_KS::id_to_weight(36)/(B2*B2) - A*B1*LBM_KS::id_to_weight(35)/((B2*B2)*B3) - 9*A*(B2*B2)*LBM_KS::id_to_weight(34) - 4*A*B2*(B3*B3)*LBM_KS::id_to_weight(33) - 4*A*B2*LBM_KS::id_to_weight(32)/(B3*B3) - A*B3*LBM_KS::id_to_weight(31) - A*LBM_KS::id_to_weight(30) - A*LBM_KS::id_to_weight(29)/B3 - A*B3*LBM_KS::id_to_weight(23)/(B2*B2) - A*LBM_KS::id_to_weight(22)/(B2*B2) - A*LBM_KS::id_to_weight(21)/((B2*B2)*B3) - 4*A*(B3*B3)*LBM_KS::id_to_weight(20)/(B2*B2*B2) - 4*A*LBM_KS::id_to_weight(19)/((B2*B2*B2)*(B3*B3)) - 9*A*LBM_KS::id_to_weight(18)/(B2*B2*B2*B2) - A*B3*LBM_KS::id_to_weight(17)/B1 - A*LBM_KS::id_to_weight(16)/B1 - A*LBM_KS::id_to_weight(15)/(B1*B3) - A*B3*LBM_KS::id_to_weight(11)/(B1*(B2*B2)) - A*LBM_KS::id_to_weight(10)/(B1*(B2*B2)) - A*LBM_KS::id_to_weight(9)/(B1*(B2*B2)*B3) - 4*A*B2*(B3*B3)*LBM_KS::id_to_weight(8)/(B1*B1) - 4*A*B2*LBM_KS::id_to_weight(7)/(B1*B1) - 4*A*B2*LBM_KS::id_to_weight(6)/((B1*B1)*(B3*B3)) - 4*A*(B3*B3)*LBM_KS::id_to_weight(3)/((B1*B1)*(B2*B2*B2)) - 4*A*LBM_KS::id_to_weight(2)/((B1*B1)*(B2*B2*B2)) - 4*A*LBM_KS::id_to_weight(1)/((B1*B1)*(B2*B2*B2)*(B3*B3));
J[2][3] = -4*A*(B1*B1)*(B2*B2)*B3*LBM_KS::id_to_weight(51) + 4*A*(B1*B1)*(B2*B2)*LBM_KS::id_to_weight(49)/(B3*B3*B3) + 4*A*(B1*B1)*B3*LBM_KS::id_to_weight(46)/(B2*B2) - 4*A*(B1*B1)*LBM_KS::id_to_weight(44)/((B2*B2)*(B3*B3*B3)) - A*B1*B2*LBM_KS::id_to_weight(43) + A*B1*B2*LBM_KS::id_to_weight(41)/(B3*B3) + A*B1*LBM_KS::id_to_weight(37)/B2 - A*B1*LBM_KS::id_to_weight(35)/(B2*(B3*B3)) - 4*A*(B2*B2)*B3*LBM_KS::id_to_weight(33) + 4*A*(B2*B2)*LBM_KS::id_to_weight(32)/(B3*B3*B3) - A*B2*LBM_KS::id_to_weight(31) + A*B2*LBM_KS::id_to_weight(29)/(B3*B3) + A*LBM_KS::id_to_weight(23)/B2 - A*LBM_KS::id_to_weight(21)/(B2*(B3*B3)) + 4*A*B3*LBM_KS::id_to_weight(20)/(B2*B2) - 4*A*LBM_KS::id_to_weight(19)/((B2*B2)*(B3*B3*B3)) - A*B2*LBM_KS::id_to_weight(17)/B1 + A*B2*LBM_KS::id_to_weight(15)/(B1*(B3*B3)) + A*LBM_KS::id_to_weight(11)/(B1*B2) - A*LBM_KS::id_to_weight(9)/(B1*B2*(B3*B3)) - 4*A*(B2*B2)*B3*LBM_KS::id_to_weight(8)/(B1*B1) + 4*A*(B2*B2)*LBM_KS::id_to_weight(6)/((B1*B1)*(B3*B3*B3)) + 4*A*B3*LBM_KS::id_to_weight(3)/((B1*B1)*(B2*B2)) - 4*A*LBM_KS::id_to_weight(1)/((B1*B1)*(B2*B2)*(B3*B3*B3));
J[3][0] = -2*(B1*B1)*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(51) + 2*(B1*B1)*(B2*B2)*LBM_KS::id_to_weight(49)/(B3*B3) - 2*(B1*B1)*(B3*B3)*LBM_KS::id_to_weight(48) + 2*(B1*B1)*LBM_KS::id_to_weight(47)/(B3*B3) - 2*(B1*B1)*(B3*B3)*LBM_KS::id_to_weight(46)/(B2*B2) + 2*(B1*B1)*LBM_KS::id_to_weight(44)/((B2*B2)*(B3*B3)) - B1*B2*B3*LBM_KS::id_to_weight(43) + B1*B2*LBM_KS::id_to_weight(41)/B3 - B1*B3*LBM_KS::id_to_weight(40) + B1*LBM_KS::id_to_weight(38)/B3 - B1*B3*LBM_KS::id_to_weight(37)/B2 + B1*LBM_KS::id_to_weight(35)/(B2*B3) - 2*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(33) + 2*(B2*B2)*LBM_KS::id_to_weight(32)/(B3*B3) - B2*B3*LBM_KS::id_to_weight(31) + B2*LBM_KS::id_to_weight(29)/B3 - 3*(B3*B3*B3)*LBM_KS::id_to_weight(28) - B3*LBM_KS::id_to_weight(27) + LBM_KS::id_to_weight(25)/B3 + 3*LBM_KS::id_to_weight(24)/(B3*B3*B3) - B3*LBM_KS::id_to_weight(23)/B2 + LBM_KS::id_to_weight(21)/(B2*B3) - 2*(B3*B3)*LBM_KS::id_to_weight(20)/(B2*B2) + 2*LBM_KS::id_to_weight(19)/((B2*B2)*(B3*B3)) - B2*B3*LBM_KS::id_to_weight(17)/B1 + B2*LBM_KS::id_to_weight(15)/(B1*B3) - B3*LBM_KS::id_to_weight(14)/B1 + LBM_KS::id_to_weight(12)/(B1*B3) - B3*LBM_KS::id_to_weight(11)/(B1*B2) + LBM_KS::id_to_weight(9)/(B1*B2*B3) - 2*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(8)/(B1*B1) + 2*(B2*B2)*LBM_KS::id_to_weight(6)/((B1*B1)*(B3*B3)) - 2*(B3*B3)*LBM_KS::id_to_weight(5)/(B1*B1) + 2*LBM_KS::id_to_weight(4)/((B1*B1)*(B3*B3)) - 2*(B3*B3)*LBM_KS::id_to_weight(3)/((B1*B1)*(B2*B2)) + 2*LBM_KS::id_to_weight(1)/((B1*B1)*(B2*B2)*(B3*B3));
J[3][1] = -4*A*B1*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(51) + 4*A*B1*(B2*B2)*LBM_KS::id_to_weight(49)/(B3*B3) - 4*A*B1*(B3*B3)*LBM_KS::id_to_weight(48) + 4*A*B1*LBM_KS::id_to_weight(47)/(B3*B3) - 4*A*B1*(B3*B3)*LBM_KS::id_to_weight(46)/(B2*B2) + 4*A*B1*LBM_KS::id_to_weight(44)/((B2*B2)*(B3*B3)) - A*B2*B3*LBM_KS::id_to_weight(43) + A*B2*LBM_KS::id_to_weight(41)/B3 - A*B3*LBM_KS::id_to_weight(40) + A*LBM_KS::id_to_weight(38)/B3 - A*B3*LBM_KS::id_to_weight(37)/B2 + A*LBM_KS::id_to_weight(35)/(B2*B3) + A*B2*B3*LBM_KS::id_to_weight(17)/(B1*B1) - A*B2*LBM_KS::id_to_weight(15)/((B1*B1)*B3) + A*B3*LBM_KS::id_to_weight(14)/(B1*B1) - A*LBM_KS::id_to_weight(12)/((B1*B1)*B3) + A*B3*LBM_KS::id_to_weight(11)/((B1*B1)*B2) - A*LBM_KS::id_to_weight(9)/((B1*B1)*B2*B3) + 4*A*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(8)/(B1*B1*B1) - 4*A*(B2*B2)*LBM_KS::id_to_weight(6)/((B1*B1*B1)*(B3*B3)) + 4*A*(B3*B3)*LBM_KS::id_to_weight(5)/(B1*B1*B1) - 4*A*LBM_KS::id_to_weight(4)/((B1*B1*B1)*(B3*B3)) + 4*A*(B3*B3)*LBM_KS::id_to_weight(3)/((B1*B1*B1)*(B2*B2)) - 4*A*LBM_KS::id_to_weight(1)/((B1*B1*B1)*(B2*B2)*(B3*B3));
J[3][2] = -4*A*(B1*B1)*B2*(B3*B3)*LBM_KS::id_to_weight(51) + 4*A*(B1*B1)*B2*LBM_KS::id_to_weight(49)/(B3*B3) + 4*A*(B1*B1)*(B3*B3)*LBM_KS::id_to_weight(46)/(B2*B2*B2) - 4*A*(B1*B1)*LBM_KS::id_to_weight(44)/((B2*B2*B2)*(B3*B3)) - A*B1*B3*LBM_KS::id_to_weight(43) + A*B1*LBM_KS::id_to_weight(41)/B3 + A*B1*B3*LBM_KS::id_to_weight(37)/(B2*B2) - A*B1*LBM_KS::id_to_weight(35)/((B2*B2)*B3) - 4*A*B2*(B3*B3)*LBM_KS::id_to_weight(33) + 4*A*B2*LBM_KS::id_to_weight(32)/(B3*B3) - A*B3*LBM_KS::id_to_weight(31) + A*LBM_KS::id_to_weight(29)/B3 + A*B3*LBM_KS::id_to_weight(23)/(B2*B2) - A*LBM_KS::id_to_weight(21)/((B2*B2)*B3) + 4*A*(B3*B3)*LBM_KS::id_to_weight(20)/(B2*B2*B2) - 4*A*LBM_KS::id_to_weight(19)/((B2*B2*B2)*(B3*B3)) - A*B3*LBM_KS::id_to_weight(17)/B1 + A*LBM_KS::id_to_weight(15)/(B1*B3) + A*B3*LBM_KS::id_to_weight(11)/(B1*(B2*B2)) - A*LBM_KS::id_to_weight(9)/(B1*(B2*B2)*B3) - 4*A*B2*(B3*B3)*LBM_KS::id_to_weight(8)/(B1*B1) + 4*A*B2*LBM_KS::id_to_weight(6)/((B1*B1)*(B3*B3)) + 4*A*(B3*B3)*LBM_KS::id_to_weight(3)/((B1*B1)*(B2*B2*B2)) - 4*A*LBM_KS::id_to_weight(1)/((B1*B1)*(B2*B2*B2)*(B3*B3));
J[3][3] = -4*A*(B1*B1)*(B2*B2)*B3*LBM_KS::id_to_weight(51) - 4*A*(B1*B1)*(B2*B2)*LBM_KS::id_to_weight(49)/(B3*B3*B3) - 4*A*(B1*B1)*B3*LBM_KS::id_to_weight(48) - 4*A*(B1*B1)*LBM_KS::id_to_weight(47)/(B3*B3*B3) - 4*A*(B1*B1)*B3*LBM_KS::id_to_weight(46)/(B2*B2) - 4*A*(B1*B1)*LBM_KS::id_to_weight(44)/((B2*B2)*(B3*B3*B3)) - A*B1*B2*LBM_KS::id_to_weight(43) - A*B1*B2*LBM_KS::id_to_weight(41)/(B3*B3) - A*B1*LBM_KS::id_to_weight(40) - A*B1*LBM_KS::id_to_weight(38)/(B3*B3) - A*B1*LBM_KS::id_to_weight(37)/B2 - A*B1*LBM_KS::id_to_weight(35)/(B2*(B3*B3)) - 4*A*(B2*B2)*B3*LBM_KS::id_to_weight(33) - 4*A*(B2*B2)*LBM_KS::id_to_weight(32)/(B3*B3*B3) - A*B2*LBM_KS::id_to_weight(31) - A*B2*LBM_KS::id_to_weight(29)/(B3*B3) - 9*A*(B3*B3)*LBM_KS::id_to_weight(28) - A*LBM_KS::id_to_weight(27) - A*LBM_KS::id_to_weight(25)/(B3*B3) - 9*A*LBM_KS::id_to_weight(24)/(B3*B3*B3*B3) - A*LBM_KS::id_to_weight(23)/B2 - A*LBM_KS::id_to_weight(21)/(B2*(B3*B3)) - 4*A*B3*LBM_KS::id_to_weight(20)/(B2*B2) - 4*A*LBM_KS::id_to_weight(19)/((B2*B2)*(B3*B3*B3)) - A*B2*LBM_KS::id_to_weight(17)/B1 - A*B2*LBM_KS::id_to_weight(15)/(B1*(B3*B3)) - A*LBM_KS::id_to_weight(14)/B1 - A*LBM_KS::id_to_weight(12)/(B1*(B3*B3)) - A*LBM_KS::id_to_weight(11)/(B1*B2) - A*LBM_KS::id_to_weight(9)/(B1*B2*(B3*B3)) - 4*A*(B2*B2)*B3*LBM_KS::id_to_weight(8)/(B1*B1) - 4*A*(B2*B2)*LBM_KS::id_to_weight(6)/((B1*B1)*(B3*B3*B3)) - 4*A*B3*LBM_KS::id_to_weight(5)/(B1*B1) - 4*A*LBM_KS::id_to_weight(4)/((B1*B1)*(B3*B3*B3)) - 4*A*B3*LBM_KS::id_to_weight(3)/((B1*B1)*(B2*B2)) - 4*A*LBM_KS::id_to_weight(1)/((B1*B1)*(B2*B2)*(B3*B3*B3));
		}
        // Inverse method 1, Adj matrix
        dreal inverseJ[4][4];
        {
            const dreal t3  = J[0][0]*J[1][1]*J[2][3]*J[3][2];
            const dreal t2  = J[0][0]*J[1][1]*J[2][2]*J[3][3];
            const dreal t4  = J[0][0]*J[1][2]*J[2][1]*J[3][3];
            const dreal t5  = J[0][0]*J[1][2]*J[2][3]*J[3][1];
            const dreal t6  = J[0][0]*J[1][3]*J[2][1]*J[3][2];
            const dreal t7  = J[0][0]*J[1][3]*J[2][2]*J[3][1];
            const dreal t8  = J[0][1]*J[1][0]*J[2][2]*J[3][3];
            const dreal t9  = J[0][1]*J[1][0]*J[2][3]*J[3][2];
            const dreal t10 = J[0][1]*J[1][2]*J[2][0]*J[3][3];
            const dreal t11 = J[0][1]*J[1][2]*J[2][3]*J[3][0];
            const dreal t12 = J[0][1]*J[1][3]*J[2][0]*J[3][2];
            const dreal t13 = J[0][1]*J[1][3]*J[2][2]*J[3][0];
            const dreal t14 = J[0][2]*J[1][0]*J[2][1]*J[3][3];
            const dreal t15 = J[0][2]*J[1][0]*J[2][3]*J[3][1];
            const dreal t16 = J[0][2]*J[1][1]*J[2][0]*J[3][3];
            const dreal t17 = J[0][2]*J[1][1]*J[2][3]*J[3][0];
            const dreal t18 = J[0][2]*J[1][3]*J[2][0]*J[3][1];
            const dreal t19 = J[0][2]*J[1][3]*J[2][1]*J[3][0];
            const dreal t20 = J[0][3]*J[1][0]*J[2][1]*J[3][2];
            const dreal t21 = J[0][3]*J[1][0]*J[2][2]*J[3][1];
            const dreal t22 = J[0][3]*J[1][1]*J[2][0]*J[3][2];
            const dreal t23 = J[0][3]*J[1][1]*J[2][2]*J[3][0];
            const dreal t24 = J[0][3]*J[1][2]*J[2][0]*J[3][1];
            const dreal t25 = J[0][3]*J[1][2]*J[2][1]*J[3][0];
            const dreal t26 = -t3;
            const dreal t27 = -t4;
            const dreal t28 = -t7;
            const dreal t29 = -t8;
            const dreal t30 = -t11;
            const dreal t31 = -t12;
            const dreal t32 = -t15;
            const dreal t33 = -t16;
            const dreal t34 = -t19;
            const dreal t35 = -t20;
            const dreal t36 = -t23;
            const dreal t37 = -t24;
            const dreal t38 = t2+t5+t6+t9+t10+t13+t14+t17+t18+t21+t22+t25+t26+t27+t28+t29+t30+t31+t32+t33+t34+t35+t36+t37;
            const dreal t39 = 1.0/t38;



            inverseJ[0][0] = t39*(J[1][1]*J[2][2]*J[3][3]-J[1][1]*J[2][3]*J[3][2]-J[1][2]*J[2][1]*J[3][3]+J[1][2]*J[2][3]*J[3][1]+J[1][3]*J[2][1]*J[3][2]-J[1][3]*J[2][2]*J[3][1]);
            inverseJ[0][1] = -t39*(J[0][1]*J[2][2]*J[3][3]-J[0][1]*J[2][3]*J[3][2]-J[0][2]*J[2][1]*J[3][3]+J[0][2]*J[2][3]*J[3][1]+J[0][3]*J[2][1]*J[3][2]-J[0][3]*J[2][2]*J[3][1]);
            inverseJ[0][2] = t39*(J[0][1]*J[1][2]*J[3][3]-J[0][1]*J[1][3]*J[3][2]-J[0][2]*J[1][1]*J[3][3]+J[0][2]*J[1][3]*J[3][1]+J[0][3]*J[1][1]*J[3][2]-J[0][3]*J[1][2]*J[3][1]);
            inverseJ[0][3] = -t39*(J[0][1]*J[1][2]*J[2][3]-J[0][1]*J[1][3]*J[2][2]-J[0][2]*J[1][1]*J[2][3]+J[0][2]*J[1][3]*J[2][1]+J[0][3]*J[1][1]*J[2][2]-J[0][3]*J[1][2]*J[2][1]);
            inverseJ[1][0] = -t39*(J[1][0]*J[2][2]*J[3][3]-J[1][0]*J[2][3]*J[3][2]-J[1][2]*J[2][0]*J[3][3]+J[1][2]*J[2][3]*J[3][0]+J[1][3]*J[2][0]*J[3][2]-J[1][3]*J[2][2]*J[3][0]);
            inverseJ[1][1] = t39*(J[0][0]*J[2][2]*J[3][3]-J[0][0]*J[2][3]*J[3][2]-J[0][2]*J[2][0]*J[3][3]+J[0][2]*J[2][3]*J[3][0]+J[0][3]*J[2][0]*J[3][2]-J[0][3]*J[2][2]*J[3][0]);
            inverseJ[1][2] = -t39*(J[0][0]*J[1][2]*J[3][3]-J[0][0]*J[1][3]*J[3][2]-J[0][2]*J[1][0]*J[3][3]+J[0][2]*J[1][3]*J[3][0]+J[0][3]*J[1][0]*J[3][2]-J[0][3]*J[1][2]*J[3][0]);
            inverseJ[1][3] = t39*(J[0][0]*J[1][2]*J[2][3]-J[0][0]*J[1][3]*J[2][2]-J[0][2]*J[1][0]*J[2][3]+J[0][2]*J[1][3]*J[2][0]+J[0][3]*J[1][0]*J[2][2]-J[0][3]*J[1][2]*J[2][0]);
            inverseJ[2][0] = t39*(J[1][0]*J[2][1]*J[3][3]-J[1][0]*J[2][3]*J[3][1]-J[1][1]*J[2][0]*J[3][3]+J[1][1]*J[2][3]*J[3][0]+J[1][3]*J[2][0]*J[3][1]-J[1][3]*J[2][1]*J[3][0]);
            inverseJ[2][1] = -t39*(J[0][0]*J[2][1]*J[3][3]-J[0][0]*J[2][3]*J[3][1]-J[0][1]*J[2][0]*J[3][3]+J[0][1]*J[2][3]*J[3][0]+J[0][3]*J[2][0]*J[3][1]-J[0][3]*J[2][1]*J[3][0]);
            inverseJ[2][2] = t39*(J[0][0]*J[1][1]*J[3][3]-J[0][0]*J[1][3]*J[3][1]-J[0][1]*J[1][0]*J[3][3]+J[0][1]*J[1][3]*J[3][0]+J[0][3]*J[1][0]*J[3][1]-J[0][3]*J[1][1]*J[3][0]);
            inverseJ[2][3] = -t39*(J[0][0]*J[1][1]*J[2][3]-J[0][0]*J[1][3]*J[2][1]-J[0][1]*J[1][0]*J[2][3]+J[0][1]*J[1][3]*J[2][0]+J[0][3]*J[1][0]*J[2][1]-J[0][3]*J[1][1]*J[2][0]);
            inverseJ[3][0] = -t39*(J[1][0]*J[2][1]*J[3][2]-J[1][0]*J[2][2]*J[3][1]-J[1][1]*J[2][0]*J[3][2]+J[1][1]*J[2][2]*J[3][0]+J[1][2]*J[2][0]*J[3][1]-J[1][2]*J[2][1]*J[3][0]);
            inverseJ[3][1] = t39*(J[0][0]*J[2][1]*J[3][2]-J[0][0]*J[2][2]*J[3][1]-J[0][1]*J[2][0]*J[3][2]+J[0][1]*J[2][2]*J[3][0]+J[0][2]*J[2][0]*J[3][1]-J[0][2]*J[2][1]*J[3][0]);
            inverseJ[3][2] = -t39*(J[0][0]*J[1][1]*J[3][2]-J[0][0]*J[1][2]*J[3][1]-J[0][1]*J[1][0]*J[3][2]+J[0][1]*J[1][2]*J[3][0]+J[0][2]*J[1][0]*J[3][1]-J[0][2]*J[1][1]*J[3][0]);
            inverseJ[3][3] = t39*(J[0][0]*J[1][1]*J[2][2]-J[0][0]*J[1][2]*J[2][1]-J[0][1]*J[1][0]*J[2][2]+J[0][1]*J[1][2]*J[2][0]+J[0][2]*J[1][0]*J[2][1]-J[0][2]*J[1][1]*J[2][0]);
        }
        // Newton method step
        A  = A + (- inverseJ[0][0]*F[0] - inverseJ[0][1]*F[1] - inverseJ[0][2]*F[2] - inverseJ[0][3]*F[3]);
        B1 = B1 + (- inverseJ[1][0]*F[0] - inverseJ[1][1]*F[1] - inverseJ[1][2]*F[2] - inverseJ[1][3]*F[3]);
        B2 = B2 + (- inverseJ[2][0]*F[0] - inverseJ[2][1]*F[1] - inverseJ[2][2]*F[2] - inverseJ[2][3]*F[3]);
        B3 = B3 + (- inverseJ[3][0]*F[0] - inverseJ[3][1]*F[1] - inverseJ[3][2]*F[2] - inverseJ[3][3]*F[3]);

        // Inverse method 2, symbolic LU decomposition
        /*
        const dreal Ux_0 =  F[0];
        const dreal Ux_1 = -F[0]*J[1][0]/J[0][0] + F[1];
        const dreal Ux_2 = -F[0]*J[2][0]/J[0][0] + F[2] - (J[2][1] - J[0][1]*J[2][0]/J[0][0])*(-F[0]*J[1][0]/J[0][0] + F[1])/(J[1][1] - J[0][1]*J[1][0]/J[0][0]);
        const dreal Ux_3 = -F[0]*J[3][0]/J[0][0] + F[3] - (J[3][2] - (J[1][2] - J[0][2]*J[1][0]/J[0][0])*(J[3][1] - J[0][1]*J[3][0]/J[0][0])/(J[1][1] - J[0][1]*J[1][0]/J[0][0]) - J[0][2]*J[3][0]/J[0][0])*(-F[0]*J[2][0]/J[0][0] + F[2] - (J[2][1] - J[0][1]*J[2][0]/J[0][0])*(-F[0]*J[1][0]/J[0][0] + F[1])/(J[1][1] - J[0][1]*J[1][0]/J[0][0]))/(J[2][2] - (J[1][2] - J[0][2]*J[1][0]/J[0][0])*(J[2][1] - J[0][1]*J[2][0]/J[0][0])/(J[1][1] - J[0][1]*J[1][0]/J[0][0]) - J[0][2]*J[2][0]/J[0][0]) - (J[3][1] - J[0][1]*J[3][0]/J[0][0])*(-F[0]*J[1][0]/J[0][0] + F[1])/(J[1][1] - J[0][1]*J[1][0]/J[0][0]);
        const dreal x_1 = Ux_0/J[0][0];
        const dreal x_2 = Ux_1/(J[1][1] - J[0][1]*J[1][0]/J[0][0]);
        const dreal x_3 = Ux_2/(J[2][2] - (J[1][2] - J[0][2]*J[1][0]/J[0][0])*(J[2][1] - J[0][1]*J[2][0]/J[0][0])/(J[1][1] - J[0][1]*J[1][0]/J[0][0]) - J[0][2]*J[2][0]/J[0][0]);
        const dreal x_4 = Ux_3/(J[3][3] - (J[2][3] - (J[1][3] - J[0][3]*J[1][0]/J[0][0])*(J[2][1] - J[0][1]*J[2][0]/J[0][0])/(J[1][1] - J[0][1]*J[1][0]/J[0][0]) - J[0][3]*J[2][0]/J[0][0])*(J[3][2] - (J[1][2] - J[0][2]*J[1][0]/J[0][0])*(J[3][1] - J[0][1]*J[3][0]/J[0][0])/(J[1][1] - J[0][1]*J[1][0]/J[0][0]) - J[0][2]*J[3][0]/J[0][0])/(J[2][2] - (J[1][2] - J[0][2]*J[1][0]/J[0][0])*(J[2][1] - J[0][1]*J[2][0]/J[0][0])/(J[1][1] - J[0][1]*J[1][0]/J[0][0]) - J[0][2]*J[2][0]/J[0][0]) - (J[1][3] - J[0][3]*J[1][0]/J[0][0])*(J[3][1] - J[0][1]*J[3][0]/J[0][0])/(J[1][1] - J[0][1]*J[1][0]/J[0][0]) - J[0][3]*J[3][0]/J[0][0]);

        A = A - x_1;
        B1 = B1 - x_2;
        B2 = B2 - x_3;
        B3 = B3 - x_4;
        */
    }

    template <typename LBM_KS>
    __cuda_callable__ static void calculateConservationLawsRightHandSide(dreal (&F)[4], dreal &A, dreal &B1, dreal &B2, dreal &B3, LBM_KS &KS){
F[0] = -A*(B1*B1*B1)*LBM_KS::id_to_weight(52) - A*(B1*B1)*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(51) - A*(B1*B1)*(B2*B2)*LBM_KS::id_to_weight(50) - A*(B1*B1)*(B2*B2)*LBM_KS::id_to_weight(49)/(B3*B3) - A*(B1*B1)*(B3*B3)*LBM_KS::id_to_weight(48) - A*(B1*B1)*LBM_KS::id_to_weight(47)/(B3*B3) - A*(B1*B1)*(B3*B3)*LBM_KS::id_to_weight(46)/(B2*B2) - A*(B1*B1)*LBM_KS::id_to_weight(45)/(B2*B2) - A*(B1*B1)*LBM_KS::id_to_weight(44)/((B2*B2)*(B3*B3)) - A*B1*B2*B3*LBM_KS::id_to_weight(43) - A*B1*B2*LBM_KS::id_to_weight(42) - A*B1*B2*LBM_KS::id_to_weight(41)/B3 - A*B1*B3*LBM_KS::id_to_weight(40) - A*B1*LBM_KS::id_to_weight(39) - A*B1*LBM_KS::id_to_weight(38)/B3 - A*B1*B3*LBM_KS::id_to_weight(37)/B2 - A*B1*LBM_KS::id_to_weight(36)/B2 - A*B1*LBM_KS::id_to_weight(35)/(B2*B3) - A*(B2*B2*B2)*LBM_KS::id_to_weight(34) - A*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(33) - A*(B2*B2)*LBM_KS::id_to_weight(32)/(B3*B3) - A*B2*B3*LBM_KS::id_to_weight(31) - A*B2*LBM_KS::id_to_weight(30) - A*B2*LBM_KS::id_to_weight(29)/B3 - A*(B3*B3*B3)*LBM_KS::id_to_weight(28) - A*B3*LBM_KS::id_to_weight(27) - A*LBM_KS::id_to_weight(26) - A*LBM_KS::id_to_weight(25)/B3 - A*LBM_KS::id_to_weight(24)/(B3*B3*B3) - A*B3*LBM_KS::id_to_weight(23)/B2 - A*LBM_KS::id_to_weight(22)/B2 - A*LBM_KS::id_to_weight(21)/(B2*B3) - A*(B3*B3)*LBM_KS::id_to_weight(20)/(B2*B2) - A*LBM_KS::id_to_weight(19)/((B2*B2)*(B3*B3)) - A*LBM_KS::id_to_weight(18)/(B2*B2*B2) - A*B2*B3*LBM_KS::id_to_weight(17)/B1 - A*B2*LBM_KS::id_to_weight(16)/B1 - A*B2*LBM_KS::id_to_weight(15)/(B1*B3) - A*B3*LBM_KS::id_to_weight(14)/B1 - A*LBM_KS::id_to_weight(13)/B1 - A*LBM_KS::id_to_weight(12)/(B1*B3) - A*B3*LBM_KS::id_to_weight(11)/(B1*B2) - A*LBM_KS::id_to_weight(10)/(B1*B2) - A*LBM_KS::id_to_weight(9)/(B1*B2*B3) - A*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(8)/(B1*B1) - A*(B2*B2)*LBM_KS::id_to_weight(7)/(B1*B1) - A*(B2*B2)*LBM_KS::id_to_weight(6)/((B1*B1)*(B3*B3)) - A*(B3*B3)*LBM_KS::id_to_weight(5)/(B1*B1) - A*LBM_KS::id_to_weight(4)/((B1*B1)*(B3*B3)) - A*(B3*B3)*LBM_KS::id_to_weight(3)/((B1*B1)*(B2*B2)) - A*LBM_KS::id_to_weight(2)/((B1*B1)*(B2*B2)) - A*LBM_KS::id_to_weight(1)/((B1*B1)*(B2*B2)*(B3*B3)) - A*LBM_KS::id_to_weight(0)/(B1*B1*B1) + KS.rho;
F[1] = -3*A*(B1*B1*B1)*LBM_KS::id_to_weight(52) - 2*A*(B1*B1)*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(51) - 2*A*(B1*B1)*(B2*B2)*LBM_KS::id_to_weight(50) - 2*A*(B1*B1)*(B2*B2)*LBM_KS::id_to_weight(49)/(B3*B3) - 2*A*(B1*B1)*(B3*B3)*LBM_KS::id_to_weight(48) - 2*A*(B1*B1)*LBM_KS::id_to_weight(47)/(B3*B3) - 2*A*(B1*B1)*(B3*B3)*LBM_KS::id_to_weight(46)/(B2*B2) - 2*A*(B1*B1)*LBM_KS::id_to_weight(45)/(B2*B2) - 2*A*(B1*B1)*LBM_KS::id_to_weight(44)/((B2*B2)*(B3*B3)) - A*B1*B2*B3*LBM_KS::id_to_weight(43) - A*B1*B2*LBM_KS::id_to_weight(42) - A*B1*B2*LBM_KS::id_to_weight(41)/B3 - A*B1*B3*LBM_KS::id_to_weight(40) - A*B1*LBM_KS::id_to_weight(39) - A*B1*LBM_KS::id_to_weight(38)/B3 - A*B1*B3*LBM_KS::id_to_weight(37)/B2 - A*B1*LBM_KS::id_to_weight(36)/B2 - A*B1*LBM_KS::id_to_weight(35)/(B2*B3) + A*B2*B3*LBM_KS::id_to_weight(17)/B1 + A*B2*LBM_KS::id_to_weight(16)/B1 + A*B2*LBM_KS::id_to_weight(15)/(B1*B3) + A*B3*LBM_KS::id_to_weight(14)/B1 + A*LBM_KS::id_to_weight(13)/B1 + A*LBM_KS::id_to_weight(12)/(B1*B3) + A*B3*LBM_KS::id_to_weight(11)/(B1*B2) + A*LBM_KS::id_to_weight(10)/(B1*B2) + A*LBM_KS::id_to_weight(9)/(B1*B2*B3) + 2*A*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(8)/(B1*B1) + 2*A*(B2*B2)*LBM_KS::id_to_weight(7)/(B1*B1) + 2*A*(B2*B2)*LBM_KS::id_to_weight(6)/((B1*B1)*(B3*B3)) + 2*A*(B3*B3)*LBM_KS::id_to_weight(5)/(B1*B1) + 2*A*LBM_KS::id_to_weight(4)/((B1*B1)*(B3*B3)) + 2*A*(B3*B3)*LBM_KS::id_to_weight(3)/((B1*B1)*(B2*B2)) + 2*A*LBM_KS::id_to_weight(2)/((B1*B1)*(B2*B2)) + 2*A*LBM_KS::id_to_weight(1)/((B1*B1)*(B2*B2)*(B3*B3)) + 3*A*LBM_KS::id_to_weight(0)/(B1*B1*B1) + KS.rho*KS.vx;
F[2] = -2*A*(B1*B1)*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(51) - 2*A*(B1*B1)*(B2*B2)*LBM_KS::id_to_weight(50) - 2*A*(B1*B1)*(B2*B2)*LBM_KS::id_to_weight(49)/(B3*B3) + 2*A*(B1*B1)*(B3*B3)*LBM_KS::id_to_weight(46)/(B2*B2) + 2*A*(B1*B1)*LBM_KS::id_to_weight(45)/(B2*B2) + 2*A*(B1*B1)*LBM_KS::id_to_weight(44)/((B2*B2)*(B3*B3)) - A*B1*B2*B3*LBM_KS::id_to_weight(43) - A*B1*B2*LBM_KS::id_to_weight(42) - A*B1*B2*LBM_KS::id_to_weight(41)/B3 + A*B1*B3*LBM_KS::id_to_weight(37)/B2 + A*B1*LBM_KS::id_to_weight(36)/B2 + A*B1*LBM_KS::id_to_weight(35)/(B2*B3) - 3*A*(B2*B2*B2)*LBM_KS::id_to_weight(34) - 2*A*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(33) - 2*A*(B2*B2)*LBM_KS::id_to_weight(32)/(B3*B3) - A*B2*B3*LBM_KS::id_to_weight(31) - A*B2*LBM_KS::id_to_weight(30) - A*B2*LBM_KS::id_to_weight(29)/B3 + A*B3*LBM_KS::id_to_weight(23)/B2 + A*LBM_KS::id_to_weight(22)/B2 + A*LBM_KS::id_to_weight(21)/(B2*B3) + 2*A*(B3*B3)*LBM_KS::id_to_weight(20)/(B2*B2) + 2*A*LBM_KS::id_to_weight(19)/((B2*B2)*(B3*B3)) + 3*A*LBM_KS::id_to_weight(18)/(B2*B2*B2) - A*B2*B3*LBM_KS::id_to_weight(17)/B1 - A*B2*LBM_KS::id_to_weight(16)/B1 - A*B2*LBM_KS::id_to_weight(15)/(B1*B3) + A*B3*LBM_KS::id_to_weight(11)/(B1*B2) + A*LBM_KS::id_to_weight(10)/(B1*B2) + A*LBM_KS::id_to_weight(9)/(B1*B2*B3) - 2*A*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(8)/(B1*B1) - 2*A*(B2*B2)*LBM_KS::id_to_weight(7)/(B1*B1) - 2*A*(B2*B2)*LBM_KS::id_to_weight(6)/((B1*B1)*(B3*B3)) + 2*A*(B3*B3)*LBM_KS::id_to_weight(3)/((B1*B1)*(B2*B2)) + 2*A*LBM_KS::id_to_weight(2)/((B1*B1)*(B2*B2)) + 2*A*LBM_KS::id_to_weight(1)/((B1*B1)*(B2*B2)*(B3*B3)) + KS.rho*KS.vy;
F[3] = -2*A*(B1*B1)*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(51) + 2*A*(B1*B1)*(B2*B2)*LBM_KS::id_to_weight(49)/(B3*B3) - 2*A*(B1*B1)*(B3*B3)*LBM_KS::id_to_weight(48) + 2*A*(B1*B1)*LBM_KS::id_to_weight(47)/(B3*B3) - 2*A*(B1*B1)*(B3*B3)*LBM_KS::id_to_weight(46)/(B2*B2) + 2*A*(B1*B1)*LBM_KS::id_to_weight(44)/((B2*B2)*(B3*B3)) - A*B1*B2*B3*LBM_KS::id_to_weight(43) + A*B1*B2*LBM_KS::id_to_weight(41)/B3 - A*B1*B3*LBM_KS::id_to_weight(40) + A*B1*LBM_KS::id_to_weight(38)/B3 - A*B1*B3*LBM_KS::id_to_weight(37)/B2 + A*B1*LBM_KS::id_to_weight(35)/(B2*B3) - 2*A*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(33) + 2*A*(B2*B2)*LBM_KS::id_to_weight(32)/(B3*B3) - A*B2*B3*LBM_KS::id_to_weight(31) + A*B2*LBM_KS::id_to_weight(29)/B3 - 3*A*(B3*B3*B3)*LBM_KS::id_to_weight(28) - A*B3*LBM_KS::id_to_weight(27) + A*LBM_KS::id_to_weight(25)/B3 + 3*A*LBM_KS::id_to_weight(24)/(B3*B3*B3) - A*B3*LBM_KS::id_to_weight(23)/B2 + A*LBM_KS::id_to_weight(21)/(B2*B3) - 2*A*(B3*B3)*LBM_KS::id_to_weight(20)/(B2*B2) + 2*A*LBM_KS::id_to_weight(19)/((B2*B2)*(B3*B3)) - A*B2*B3*LBM_KS::id_to_weight(17)/B1 + A*B2*LBM_KS::id_to_weight(15)/(B1*B3) - A*B3*LBM_KS::id_to_weight(14)/B1 + A*LBM_KS::id_to_weight(12)/(B1*B3) - A*B3*LBM_KS::id_to_weight(11)/(B1*B2) + A*LBM_KS::id_to_weight(9)/(B1*B2*B3) - 2*A*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(8)/(B1*B1) + 2*A*(B2*B2)*LBM_KS::id_to_weight(6)/((B1*B1)*(B3*B3)) - 2*A*(B3*B3)*LBM_KS::id_to_weight(5)/(B1*B1) + 2*A*LBM_KS::id_to_weight(4)/((B1*B1)*(B3*B3)) - 2*A*(B3*B3)*LBM_KS::id_to_weight(3)/((B1*B1)*(B2*B2)) + 2*A*LBM_KS::id_to_weight(1)/((B1*B1)*(B2*B2)*(B3*B3)) + KS.rho*KS.vz;
}

    template <typename LBM_KS>
    __cuda_callable__ static void calculateEquilibriumDistribution(dreal (&feq)[LBM_KS::Q], dreal &A, dreal &B1, dreal &B2, dreal &B3, LBM_KS &KS){

        {
        feq[0] = A*LBM_KS::id_to_weight(0)/(B1*B1*B1);
		feq[1] = A*LBM_KS::id_to_weight(1)/((B1*B1)*(B2*B2)*(B3*B3));
		feq[2] = A*LBM_KS::id_to_weight(2)/((B1*B1)*(B2*B2));
		feq[3] = A*(B3*B3)*LBM_KS::id_to_weight(3)/((B1*B1)*(B2*B2));
		feq[4] = A*LBM_KS::id_to_weight(4)/((B1*B1)*(B3*B3));
		feq[5] = A*(B3*B3)*LBM_KS::id_to_weight(5)/(B1*B1);
		feq[6] = A*(B2*B2)*LBM_KS::id_to_weight(6)/((B1*B1)*(B3*B3));
		feq[7] = A*(B2*B2)*LBM_KS::id_to_weight(7)/(B1*B1);
		feq[8] = A*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(8)/(B1*B1);
		feq[9] = A*LBM_KS::id_to_weight(9)/(B1*B2*B3);
		feq[10] = A*LBM_KS::id_to_weight(10)/(B1*B2);
		feq[11] = A*B3*LBM_KS::id_to_weight(11)/(B1*B2);
		feq[12] = A*LBM_KS::id_to_weight(12)/(B1*B3);
		feq[13] = A*LBM_KS::id_to_weight(13)/B1;
		feq[14] = A*B3*LBM_KS::id_to_weight(14)/B1;
		feq[15] = A*B2*LBM_KS::id_to_weight(15)/(B1*B3);
		feq[16] = A*B2*LBM_KS::id_to_weight(16)/B1;
		feq[17] = A*B2*B3*LBM_KS::id_to_weight(17)/B1;
		feq[18] = A*LBM_KS::id_to_weight(18)/(B2*B2*B2);
		feq[19] = A*LBM_KS::id_to_weight(19)/((B2*B2)*(B3*B3));
		feq[20] = A*(B3*B3)*LBM_KS::id_to_weight(20)/(B2*B2);
		feq[21] = A*LBM_KS::id_to_weight(21)/(B2*B3);
		feq[22] = A*LBM_KS::id_to_weight(22)/B2;
		feq[23] = A*B3*LBM_KS::id_to_weight(23)/B2;
		feq[24] = A*LBM_KS::id_to_weight(24)/(B3*B3*B3);
		feq[25] = A*LBM_KS::id_to_weight(25)/B3;
		feq[26] = A*LBM_KS::id_to_weight(26);
		feq[27] = A*B3*LBM_KS::id_to_weight(27);
		feq[28] = A*(B3*B3*B3)*LBM_KS::id_to_weight(28);
		feq[29] = A*B2*LBM_KS::id_to_weight(29)/B3;
		feq[30] = A*B2*LBM_KS::id_to_weight(30);
		feq[31] = A*B2*B3*LBM_KS::id_to_weight(31);
		feq[32] = A*(B2*B2)*LBM_KS::id_to_weight(32)/(B3*B3);
		feq[33] = A*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(33);
		feq[34] = A*(B2*B2*B2)*LBM_KS::id_to_weight(34);
		feq[35] = A*B1*LBM_KS::id_to_weight(35)/(B2*B3);
		feq[36] = A*B1*LBM_KS::id_to_weight(36)/B2;
		feq[37] = A*B1*B3*LBM_KS::id_to_weight(37)/B2;
		feq[38] = A*B1*LBM_KS::id_to_weight(38)/B3;
		feq[39] = A*B1*LBM_KS::id_to_weight(39);
		feq[40] = A*B1*B3*LBM_KS::id_to_weight(40);
		feq[41] = A*B1*B2*LBM_KS::id_to_weight(41)/B3;
		feq[42] = A*B1*B2*LBM_KS::id_to_weight(42);
		feq[43] = A*B1*B2*B3*LBM_KS::id_to_weight(43);
		feq[44] = A*(B1*B1)*LBM_KS::id_to_weight(44)/((B2*B2)*(B3*B3));
		feq[45] = A*(B1*B1)*LBM_KS::id_to_weight(45)/(B2*B2);
		feq[46] = A*(B1*B1)*(B3*B3)*LBM_KS::id_to_weight(46)/(B2*B2);
		feq[47] = A*(B1*B1)*LBM_KS::id_to_weight(47)/(B3*B3);
		feq[48] = A*(B1*B1)*(B3*B3)*LBM_KS::id_to_weight(48);
		feq[49] = A*(B1*B1)*(B2*B2)*LBM_KS::id_to_weight(49)/(B3*B3);
		feq[50] = A*(B1*B1)*(B2*B2)*LBM_KS::id_to_weight(50);
		feq[51] = A*(B1*B1)*(B2*B2)*(B3*B3)*LBM_KS::id_to_weight(51);
		feq[52] = A*(B1*B1*B1)*LBM_KS::id_to_weight(52);
        }
    }

    template <typename LBM_KS>
    __cuda_callable__ static void setEquilibrium(LBM_KS &KS){
        collision(KS,true);
    }

    template <typename LBM_KS>
    __cuda_callable__ static void calculateEquilibriumDistribution(dreal (&feq)[LBM_KS::Q], LBM_KS &KS){
        dreal A = KS.A;
        dreal B1 = KS.B1;
        dreal B2 = KS.B2;
        dreal B3 = KS.B3;
        findEntropicEquilibrium(A, B1, B2, B3, KS);
        calculateEquilibriumDistribution(feq,A,B1,B2,B3,KS);
    }
};



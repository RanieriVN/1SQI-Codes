function [SQIRob, F, C] = SQIRobustness( sigma )
% Computes the robustness of non-instrumentality for an assemblage sigma
%
%  This function has one required argument:
%   sigma: a 4-D array, containing the members of the assemblage. The first 
%   two dimensions contain the (unnormalised) quantum states, while the
%   remaining two dimensions are (a,x), such that sigma(:,:,a,x) =
%   \sigma_a|x. 
%
%  SQIRob = SQIRobustness(sigma) is the generalized robustness of 
%  non-instrumentality for sigma, zero whenever the assemblage is 1SQI.
%
%  [SQIRob,F] = SQIRobustness(sigma) also returns the robustness witness F. 
%  F is a dB x dB x oa x ma array. The first two dimensions contain the 
%  dB x dB elements of the function F_a|x. The last two dimensions contain 
%  (a,x). If the assemblage is SQI, F is returned as the empty array []. 
%
%  [SQIRob,F,C] = SQIAssemblage(sigma) returns the lagrange 
%  multipliers C associated to the independence on Alice's outcome
%  of trace(S_al). C is a Ndet x oa array with sum_a C_(l,a) = 1.
%
%
% requires: CVX (http://cvxr.com/cvx/), QETLAB (http://www.qetlab.com)
%           genSinglePartyArray 
%           (https://github.com/paulskrzypczyk/steeringreview)
%
% author: Ranieri V. Nery
% last updated: December 21, 2017
%
% Acknowledgements: This code is inspired on steeringRobustness by Paul 
%                   Skrzypczyk and Daniel Cavalcanti, part of the programs 
%                   that accompany the paper "Quantum steering: a review 
%                   with focus on semidefinite programming", 
%                   published in Rep. Prog. Phys. 80 024001 (2017)
%
%                   The function genSinglePartyArray is also authored by 
%                   Paul Skrzypczyk and Daniel Cavalcanti and can be found 
%                   at https://github.com/paulskrzypczyk/steeringreview
%
%==========================================================================

% dB = dim. of Bob, oa = # outcomes for Alice, ma = # inputs for Alice
[dB,~,oa,ma] = size(sigma);

% Number of deterministic strategies for Alice
Ndet = oa^ma;

% Generate array containing the deterministic behaviors D(a|x,lambda)
SingleParty = genSinglePartyArray(oa,ma);

cvx_begin sdp quiet
    
    variable F(dB, dB, oa, ma) hermitian semidefinite
    variable C(oa, Ndet)
    
    maximize real(sum(reshape(F.*conj(sigma),1,[])))-1
    
    % Forces that sum_a C(a,lambda) == 1
	squeeze(sum(C,1)) == ones(1,Ndet);

    % C(a,lambda)*Id >= sum_x F(a,x)*D(a|x,lambda)
    for a=1:oa
        for l=1:Ndet
            C(a,l)*eye(dB) - sum(squeeze(F(:,:,a,:)).*permute(repmat(SingleParty(a,:,l),[dB,1,dB]),[1,3,2]),3) >= 0;
        end
    end
        
cvx_end

% Returns the non-instrumentality robustness
SQIRob = max(cvx_optval,0);

if SQIRob == 0
    F = [];
end

end


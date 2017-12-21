function [is_SQI_assemblage,S_al,F,C] = SQIAssemblage(sigma)
% SQIAssemblage Determines whether an assemblage admits has an one-sided 
% quantum instrumental (1SQI) model or not.
%
%  This function has one required argument:
%   sigma: a 4-D array, containing the members of the assemblage. The first 
%   two dimensions contain the (unnormalised) quantum states, while the
%   remaining two dimensions are (a,x), such that sigma(:,:,a,x) =
%   \sigma_a|x. 
%
%  is_SQI_assemblage = SQIAssemblage(sigma) is the indicator function for
%  1SQI assemblages. It returns 1 if the assemblage is a valid 1SQI
%  assemblage, and 0 otherwise.
%
%  [is_SQI_assemblage,S_al] = SQIAssemblage(sigma) also returns the local
%  subnormalized states of the SQI model which reproduces the assemblage 
%  sigma when the assemblage is 1SQI.
%  S_al is a dB x dB x Ndet x oa array, with Ndet = oa^ma, containing the 
%  members of the SQI model P(lambda)*rho_(a,lambda), corresponding to each 
%  deterministic strategy and each outcome for Alice. If there is no SQI 
%  model, S_al is returned as the empty array [].
%
%  [is_SQI_assemblage,S_al,F] = SQIAssemblage(sigma) also returns the
%  functional F that certifies that sigma is not SQI. In particular, F
%  is a dB x dB x oa x ma array. The first two dimensions contain the dB x
%  dB elements of the function F_a|x. The last two dimensions contain (a,x).
%  If the assemblage is SQI, F is returned as the empty array []. 
%
%  [is_SQI_assemblage,S_al,F,C] = SQIAssemblage(sigma) returns the lagrange 
%  multipliers C associated to the independence on Alice's outcome
%  of trace(S_al). C is a Ndet x oa cell array with sum_a C_(l,a) = 0.
%
%
% requires: CVX (http://cvxr.com/cvx/), QETLAB (http://www.qetlab.com)
%           genSinglePartyArray 
%           (https://github.com/paulskrzypczyk/steeringreview)
%
% author: Ranieri V. Nery
% last updated: December 21, 2017
%
% Acknowledgements: This code is based upon the function LHSAssemblage by 
%                   Paul Skrzypczyk and Daniel Cavalcanti, part of the 
%                   programs  that accompany the paper "Quantum steering: a 
%                   review with focus on semidefinite programming", 
%                   published in Rep. Prog. Phys. 80 024001 (2017). 
%                   The original code is available at 
%                   https://github.com/paulskrzypczyk/steeringreview
%
%                   The function genSinglePartyArray is also authored by 
%                   Paul Skrzypczyk and Daniel Cavalcanti and can be found 
%                   at the link above.
%
%==========================================================================

% dB = dim. of Bob, oa = # outcomes for Alice, ma = # inputs for Alice
[dB,~,oa,ma] = size(sigma);

% Number of deterministic strategies for Alice
Ndet = oa^ma; 

% Generate array containing the deterministic behaviors D(a|x,lambda)
SingleParty = genSinglePartyArray(oa,ma); 

%==========================================================================

cvx_begin sdp quiet
    
    % S_al are the members of the 1SQI model
    variable S_al(dB,dB,Ndet,oa) hermitian semidefinite
    
    dual variable F
    dual variables C{Ndet,oa}

    subject to
    
    % sig_a|x == \sum_l D(a|x,l) S_al 
    F : sigma == squeeze(sum(repmat(S_al,[1,1,1,1,ma])...
        .*permute(repmat(SingleParty,[1,1,1,dB,dB]),[4,5,3,1,2]),3));
    
    % P(lambda) == trace(S_al(:,:,lambda,a))
    for L=1:Ndet
        for a=2:oa
    		C{L,a} : trace(S_al(:,:,L,a)) == trace(S_al(:,:,L,1));
        end
    end
    
cvx_end

% Enforcement of the condition sum_a C(lambda, a) == 0
for L=1:Ndet
    S = 0;
    for a=2:oa
        S = S - C{L,a};
    end
    
    C{L,1} = S;
end
    
% CVX will return +inf if the problem is infeasible, and 0 if feasible
% this maps {+inf,0} to {0,1}
is_SQI_assemblage = 1-min(cvx_optval,1);

% if there is no 1SQI model, then S_al is returned as the empty array
if is_SQI_assemblage == 0
    S_al = [];    
else
    F = [];
end

end


    
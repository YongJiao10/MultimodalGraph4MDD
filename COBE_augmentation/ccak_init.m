function [proj,r,Wx,Wy] = ccak_init(X,Y,K)
% CCA calculate canonical correlations with K largest canonical correlation
[Ix, Nx]=size(X);
[Iy, Ny]=size(Y);

if nargin==2
    K=min(Nx,Ny);
end

if Ix~=Iy
    error('X and Y should have same number of rows.');
end


rho=Ix-1;
eps0=(1e-8)*rho;
% rho=Ix;

X=bsxfun(@minus,X,mean(X,1));
Y=bsxfun(@minus,Y,mean(Y,1));

Cxx=(X'*X);
Cyy=Y'*Y;
Cxy=X'*Y;

invCyyCyx=(Cyy+eps0*eye(Ny))\Cxy';

[Wx,r]=eigs((Cxx+eps0*eye(Nx))\Cxy*invCyyCyx,K);
Wy=invCyyCyx*Wx./rho;

r=diag(r)'.^.5;

Wx=orth(X*Wx);
Wy=orth(Y*Wy);

C=eye(2*K,2*K);
C(1:K,K+1:2*K)=Wx'*Wy;
C(K+1:2*K,1:K)=Wy'*Wx;
[u, d]=eigs(C,K,'LM');
u=u*d^.5;
proj=Wx*u(1:K,:)+Wy*u(K+1:2*K,:);

proj=orth(proj);


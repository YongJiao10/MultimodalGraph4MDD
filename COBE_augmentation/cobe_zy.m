function [Ac, Bc, f, Zc] = cobe_zy( Y,c )
% Common orthogonal basis extraction
% defopts=struct('c',[],'maxiter',2000,'PCAdim',[],'tol',1e-6,'epsilon',0.03);
% if ~exist('opts','var')
%     opts=struct;
% end
% [c maxiter PCAdim tol epsilon]=scanparam(defopts,opts);

maxiter=2000; PCAdim=[]; tol=1e-6; epsilon=0.03;

Ynrows=cellfun(@(x) size(x,1),Y);
NRows=size(Y{1},1);
if ~all(Ynrows==NRows)
    error('Y must have the same number of rows.');
end
Yncols=cellfun(@(x) size(x,2),Y);
N=numel(Yncols);

if isempty(PCAdim)
    PCAdim=Yncols;
end

U=cell(1,N);
J=zeros(1,N);


for n=1:N
    if PCAdim(n)==Yncols(n)
        [U{n}, R]=qr(Y{n},0);
        flag=abs(diag(R))>1e-6;
        U{n}=U{n}(:,flag);
    else
        U{n}=lowrankapp(Y{n},PCAdim(n),'pca');
    end
    J(n)=size(U{n},2);
end

if any(J>=NRows)
    error('Rank of Y{n} must be less than the number of rows of {Yn}. You may need to specify the value for PCAdim properly.');
end

%% Seeking the first common basis
% Ac=rand(NRows,1);
% Ac=Ac./norm(Ac);
[~,idx]=sort(J,'ascend');
Ac=ccak_init(U{idx(1)},U{idx(2)},1);

x=cell(1,N);
for it=1:maxiter
    c0=Ac;
    c1=zeros(NRows,1);
    for n=1:N
        x{n}(:,1)=U{n}'*Ac;
        c1=c1+U{n}*x{n}(:,1);
    end
    Ac=c1./norm(c1,'fro');
    if abs(c0'*Ac)>1-tol
        break;
    end
end


res(1)=0;
for n=1:N
    temp=U{n}'*Ac;
    res(1)=res(1)+1-temp'*temp;
end
res(1)=res(1)/N;


if res(1)>epsilon&&isempty(c)  %% c is not specified. 
    disp('No common basis found.');
    Ac=[];    
    Bc=Y;   
    return;
end


minJn=min(J);
if ~isempty(c)
    c=min([c,minJn]);
    res=[res inf(1,c-1)];
    Ac=[Ac zeros(NRows,c-1)];
else
    res=[res inf(1,minJn-1)];
    Ac=[Ac zeros(NRows,minJn-1)];
end

%% seeking the next common basis
for j=2:minJn    
    %% %% stopping criterion -- 1 where c is given
    if ~isempty(c)&&(j>c)
        break;
    end
    
    %% update U;
    for n=1:N
        U{n}=U{n}-(U{n}*x{n}(:,j-1))*x{n}(:,j-1)';
    end
    
    %% initialization
%     Ac(:,j)=U{1}*randn(size(U{1},2),1);
    Ac(:,j)=ccak_init(U{idx(1)},U{idx(2)},1);
    Ac(:,j)=Ac(:,j)./norm(Ac(:,j),'fro');
    
    %% main iterations
    for it=1:maxiter
        c0=Ac(:,j);
        c1=zeros(NRows,1);
        for n=1:N
            x{n}(:,j)=U{n}'*Ac(:,j);
            c1=c1+U{n}*x{n}(:,j);
        end        
                
        Ac(:,j)=c1./norm(c1,'fro');
        
        if abs(c0'*Ac(:,j))>1-tol
            break;
        end
        
    end % main iterations
    
    res(j)=0;
    for n=1:N
        temp=U{n}'*Ac(:,j);
        res(j)=res(j)+1-temp'*temp;
    end
    res(j)=res(j)./N;
     
    %% stopping criterion -- 2
    if res(j)>epsilon&&isempty(c)
        res(j)=inf;      
        break;
    end    
    
end % each j
% res
flag=isinf(res);
Ac(:,flag)=[];
[u , ~, v]=svd(Ac,0);
Ac=u*v';


if nargout>=2
    Bc=cell(1,N);
    Zc=cell(1,N);
    for n=1:N
        Bc{n}=(Ac'*Y{n});
        Zc{n}=Y{n}\Ac;
    end
    
    f=zeros(1,size(Ac,2));
    for j=1:size(Ac,2)
        for n=1:N
            f(j)=f(j)+norm(Y{n}*Zc{n}(:,j)-Ac(:,j))^2;
        end
    end
    f=f./N;
end

end


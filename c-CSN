function cndm = condition_ndm(data,alpha,boxsize,kk)
%Construction of conditional cell-specific network (CCSN) and conditional network degree matrix
%The function performs the transformation from gene expression matrix to
%conditional network degree matrix (cndm). 
%data: Gene expression matrix, rows = genes, columns = cells
%alpha: Significant level (eg. 0.001, 0.01, 0.05 ...)
%       larger alpha leads to more edges, Default = 0.01
%boxsize: Size of neighborhood, the value between 1 to 2 is recommended,
%kk: the number of conditional gene. when kk=0, the method is CSN
%ncell: the minmum number of 

%Define the neighborhood of each plot
[n1,n2] = size(data);
upbound = zeros(n1,n2);
lowbound = zeros(n1,n2);
for i = 1 : n1
    [s1,s2] = sort(data(i,:));
    n3 = n2-sum(sign(s1));
    h = round(boxsize*sqrt(sum(sign(s1)))); 
    k = 1;
    while k <= n2
        s = 0;
        while k+s+1 <= n2 && s1(k+s+1) == s1(k)
            s = s+1;
        end
        if s >= h
            upbound(i,s2(k:k+s)) = data(i,s2(k));
            lowbound(i,s2(k:k+s)) = data(i,s2(k));
        else
            upbound(i,s2(k:k+s)) = data(i,s2(min(n2,k+s+h)));
            lowbound(i,s2(k:k+s)) = data(i,s2(max(n3*(n3>h)+1,k-h)));
        end
        k = k+s+1;
    end
end

%Construction of CSN and network degree matrix

B = zeros(n1,n2);
cndm = zeros(n1,n2);
p = -icdf('norm',alpha,0,1);
for k = 1 : n2
    for j = 1 : n2
        B(:,j) = (data(:,j) <= upbound(:,k) & data(:,j) >= lowbound(:,k)) & data(:,k);
    end
    a = sum(B,2);
    c = B*B';
    adjmc = (c*n2-a*a')./sqrt((a*a').*((n2-a)*(n2-a)')/(n2-1)+eps);  
    adjmc = (adjmc > p);
    %kk
    if kk ~=0 
    id = condition_g(adjmc,kk);
    adjmc = zeros(n1,n1);
    for m = 1:kk   
    B_z = bsxfun(@times,B(id(m),:),B);
    idc = find(B(id(m),:)~=0);
    B_z = B_z(:,idc);
    [~,r] = size(B_z);
    a_z = sum(B_z,2);
    c_z = B_z*B_z';
    adjmc1 =(c_z*r-a_z*a_z')./sqrt((a_z*a_z').*((r-a_z)*(r-a_z)')/(r-1)+eps);
    adjmc1 = (adjmc1 > p);
    adjmc = adjmc + adjmc1;
    end
    else 
        kk=1;
    end
    cndm(:,k) = sum(adjmc/kk,2);
    display(k);
end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 a = mean(cndm);
 for i = 1 : n1
    cndm(i,:) = cndm(i,:)./(a+eps);
 end
  cndm = log(1+cndm); 
end

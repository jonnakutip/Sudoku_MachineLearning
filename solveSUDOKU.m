%% function solveSUDOKU()
% Input: 
% sudoku puzzle
% Output:
% Solution to suduko puzzle
% Author1:  PAVAN KUMAR JONNAKUTI
% E-mail1:  jpawan33@gmail.com 
% Author2:  SIVA SRINIVAS KOLUKULA
% E-mail:   allwayzitzme@gmail.com
% Web-link: http://jpavan.com
% Copyright @ Authors
% code : To solve given sudoku puzzle using Machine Learning. 
%%
clear all;clc;close all;
%%
load sudukoPuzzles.mat ;
load sudoku_weigths.mat ;
qidx = randperm(length(QQ),1);
q = reshape(QQ(qidx,:),9,9);
sol = q;
q = q(:)';
c = create_constraint_mask_matlab;
[m,n,r] = size(c);
c1 = permute(c,[1 3 2]);
c2 = reshape(c1,[],n);
idq = zeros(81,9);
idq(q>0,:) = full(ind2vec(q(q>0),9))';
empty_mask0 = (sum(idq,2)==0)' ;
[X,Y] = meshgrid(1:1:9) ;
for iloop =1:nnz(empty_mask0)
    empty_mask = (sum(idq,2)==0)' ;
    c3 = reshape(c2*idq,m,r,[]) ;
    out = permute(c3,[2 3 1]);
    k = permute(out,[2 1 3]) ;
    Pi = reshape(k,[],81)' ;
    P = Pi(empty_mask,:);
    
    s = zeros(81,9);
    s1 = P*l1';
    rs1 = relu(s1);
    s2 = rs1*l2';
    ss2 = softmax(s2')';
    s(empty_mask,:)= ss2;
    [val,idx] = max(s,[],2);
    [v,i] = max(val);
    idq(i,idx(i))=1;
    temp(iloop,:) = [i idx(i)];
end
q = vec2ind(idq')';
q = reshape(q,9,9) ;
%% Visualization of Sudoku Puzzle & Predicted Solution 
figure
subplot(1,2,1) ; % Puzzle
imagesc(reshape(reshape(QQ(qidx,:),9,9),9,9))
text(X(~empty_mask0),Y(~empty_mask0),num2str(sol(~empty_mask0)'),'color','k','FontWeight','bold','FontSize',14)
title('Original')
set(gca,'FontSize',14)
 
subplot(1,2,2) ; % Solution
imagesc(reshape(q,9,9))
text(X(empty_mask0),Y(empty_mask0),num2str(q(empty_mask0)'),'color','r','FontWeight','bold','FontSize',14)
text(X(~empty_mask0),Y(~empty_mask0),num2str(q(~empty_mask0)'),'color','k','FontWeight','bold','FontSize',14)
title('Predicted')
set(gca,'FontSize',14)
    


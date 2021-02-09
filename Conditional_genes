function id = condition_g(adjmc,kk)
a = sum(adjmc,2);
b = sum(adjmc~=0,2);
id1 = b(b>=5);
[T,INDEX]=sort(a(id1),'descend');
id2 = INDEX(1:kk);
id = id1(id2);
end

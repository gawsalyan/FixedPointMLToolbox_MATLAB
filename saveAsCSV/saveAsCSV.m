function saveAsCSV(net1)
%% EXPORT TRAINNED PARAM AS CSV FILE

 csvwrite('Wi',net1.Nets{1,1}.Layers{1,2}.Wi);
 csvwrite('Wf',net1.Nets{1,1}.Layers{1,2}.Wf);
 csvwrite('Wc',net1.Nets{1,1}.Layers{1,2}.Wc);
 csvwrite('Wo',net1.Nets{1,1}.Layers{1,2}.Wo);
 csvwrite('Wy',net1.Nets{1,1}.Layers{1,2}.Wy);
 csvwrite('Bi',net1.Nets{1,1}.Layers{1,2}.bi);
 csvwrite('Bf',net1.Nets{1,1}.Layers{1,2}.bf);
 csvwrite('Bc',net1.Nets{1,1}.Layers{1,2}.bc);
 csvwrite('Bo',net1.Nets{1,1}.Layers{1,2}.bo);
 csvwrite('By',net1.Nets{1,1}.Layers{1,2}.by);

 csvwrite('Wfc_1',net1.Nets{1,2}.Layers{1,2}.W);
 csvwrite('Bfc_1',net1.Nets{1,2}.Layers{1,2}.b);
 
 csvwrite('Wfc_2',net1.Nets{1,3}.Layers{1,2}.W);
 csvwrite('Bfc_2',net1.Nets{1,3}.Layers{1,2}.b);
 
 csvwrite('Wfc_3',net1.Nets{1,3}.Layers{1,3}.W);
 csvwrite('Bfc_3',net1.Nets{1,3}.Layers{1,3}.b);

 
 
end
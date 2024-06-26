classifierName = 'MVCO_trees_25Jun2012';
yr = 2014;
out_dir = ['\\SosikNAS1\IFCB_products\MVCO\class\class' num2str(yr) '_v1\'];%
%in_dir = 'http://ifcb-data.whoi.edu/mvco/';
%for V2 web services, set fea_dir = in_dir;
fea_dir = '\\SosikNAS1\IFCB_products\MVCO\features\features2014_v2\';
filelist = [];
%for day = 157:157,
%    filelist = [filelist list_day(datestr(datenum(yr,0,day),29), in_dir)];
%end;
disp('Checking for files to run')
filelist = dir([fea_dir '*.csv']);
filelist = {filelist.name}';
filelist = regexprep(filelist, '_fea_v2.csv', '')';
%filelist = regexprep(filelist, in_dir, '')';
files_done = dir([out_dir 'IFCB*class_v1.mat']);
files_done = char(files_done.name);
files_done = cellstr(files_done(:,1:end-13));
filelist2 = setdiff(filelist, files_done);
%if isequal(in_dir, fea_dir),
%    filelist2 = strcat(filelist2,'_features.csv');
%else
filelist2 = strcat(filelist2,'_fea_v2.csv');  %USER specify v1 or v2 features as appropriate
%end;
disp(['processing ' num2str(length(filelist2)) ' files'])
batch_classify( fea_dir, filelist2, out_dir, classifierName );


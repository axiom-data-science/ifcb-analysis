function [  ] = jump_class( hOBj, eventdata )
%function [  ] = class_change_amount( hOBj, eventdata, direction )
% callback function for 'jump to selected class' menu option in
% manual_classify for IFCB
% Heidi M. Sosik, Woods Hole Oceanographic Institution, March 2015

global category new_classcount MCflags

MCflags.class_jump = 1;
new_classcount = str2num(category(1:3));

robot_pressCR(1) % one carriage returns

end

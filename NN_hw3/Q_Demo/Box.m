function x_vec = Box(x,v_x,theta,v_theta, NUM_BOX)
    degree=pi/180;  

    if (x<-0.8)      box=1;
     elseif (x<0.8)  box=2;
     else            box=3;
    end %if

    if (v_x<-0.5)
    elseif (v_x<0.5) box=box+3;
    else             box=box+6;
    end %if

    if (theta<-6*degree)
    elseif (theta<-degree)  box=box+9;
    elseif (theta<0)        box=box+18;
    elseif (theta<degree)   box=box+27;
    elseif (theta<6*degree) box=box+36;
    else                    box=box+45;
    end  %if

    if (v_theta<-50*degree)
    elseif (v_theta<50*degree)  box=box+54;
    else                        box=box+108;
    end  %if

    if (x<-2.4 | x>2.4  | theta<-12*degree | theta>12*degree)
        box=-1;
    end
    x_vec = zeros(NUM_BOX, 1);
    if box ~= -1
        x_vec(box) = 1;
    end
end %if
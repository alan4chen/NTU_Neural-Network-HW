function [q_val,pre_state,pre_action,cur_state,cur_action] = get_action(x,v_x,theta,v_theta,reinf,q_val,pre_state,cur_state,pre_action,cur_action,ALPHA,BETA,GAMMA)

pre_state=cur_state;
pre_action=cur_action;    % Action: 1 is push left.  2 is push right

cur_state=get_box(x,v_x,theta,v_theta);

global p_before x_vec BOX_DIM BETAACE;

x_vec = zeros(BOX_DIM, 1);
if cur_state ~= -1
    x_vec(cur_state) = 1;
end

[reward_hat, p_before] = ACE(BETAACE, 0.8, 0, 0.95, p_before);
cur_action = ASE(1000, 0.9, reward_hat);


end
% if (pre_action ~= -1)   % Update Q value. If previous action been taken
%     if (cur_state == -1)  % Current state is failed
%         predicted_value=0;  % fail state's value is zero
%     elseif (q_val(cur_state,1)<=q_val(cur_state,2)) % Left Q<= Right Q
%         predicted_value=q_val(cur_state,2);         %  set Q to bigger one
%     else 
%         predicted_value=q_val(cur_state,1);
%     end %if
%     q_val(pre_state,pre_action)= q_val(pre_state,pre_action)+ ALPHA*(reinf+ GAMMA*predicted_value - q_val(pre_state,pre_action));

% end %if
% % Determine best action
% if ( q_val(cur_state,1) + (rand*BETA) <= q_val(cur_state,2) )
%      cur_action=2;  % push right
% else cur_action=1;  % push left
% end  %if

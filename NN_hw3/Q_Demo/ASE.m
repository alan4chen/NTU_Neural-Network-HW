function y = ASE(learn, decay, reward)
    global w x_vec e BETA
    noise = BETA * randn;
    
    y = Activation(w' * x_vec + noise);
    w = w + learn * reward * e;
    e = decay * e + (1-decay) * y * x_vec;
end

function y = Activation(x)
    if x >= 0
        y = 2; % push right
    else
        y = 1; % push left
    end
end
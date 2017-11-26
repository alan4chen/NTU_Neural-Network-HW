function y = ASE(learn, decay, reward)
    global w x_vec e BETA
    noise = BETA * randn;
    z = w' * x_vec + noise;
    y = 0;
    if z >= 0
        y = 2; % push right
    else
        y = 1; % push left
    end
    w = w + learn * reward * e;
    e = decay * e + (1-decay) * y * x_vec;
end
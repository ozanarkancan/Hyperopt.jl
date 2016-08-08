using Hyperopt

function objective(args)
	x,y = args
	return Dict("loss" => x ^ 2 + y, "status" => STATUS_OK)
end

trials = Trials()

best = fmin(objective,
    	space=[normal("x", 1.0, 1.0), uniform("y", -10.0, 10.0)],
	algo=TPESUGGEST,
	max_evals=10,
	trials = trials)

println(best)
println(valswithlosses(trials))

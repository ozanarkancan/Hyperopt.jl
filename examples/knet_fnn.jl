using GPUChecker, CUDArt

CUDArt.device(first_min_used_gpu())

isdefined(:MNIST) || include(Pkg.dir("Knet/examples/mnist.jl"))

using Knet, Hyperopt

@knet function mnist2layer(x; hidden=64)
	h = wbf(x; out=hidden, f=:relu, winit=Gaussian(0.0, 0.001))
	return wbf(h; out=10, f=:soft, winit=Gaussian(0.0, 0.001))
end

function train(f, data; losscnt=nothing, maxnorm=nothing)
	losscnt = 0.0
	cnt = 0.0
	for (x,ygold) in data
		ypred = forw(f, x)
		back(f, ygold, softloss)
		update!(f)
		losscnt += softloss(ypred, ygold); cnt +=1;
	end

	return losscnt
end

function test(f, data)
	sumloss = numloss = 0
	for (x,ygold) in data
		ypred = forw(f, x)
		sumloss += zeroone(ypred, ygold)
		numloss += 1
	end
	sumloss / numloss
end

function objective(args)
	println(args)
	hidden, lr = args
	net = compile(:mnist2layer; hidden=hidden)
	setp(net, lr=lr)

	global dtrn
	global dtst

	best = 1000
	for i=1:10
		train(net, dtrn)
		err = test(net, dtst)
		update!(net)

		if err < best
			best = err
		end
	end

	return Dict("loss" => best, "status" => STATUS_OK)
end

function main()

	batchsize = 64

	global dtrn = minibatch(MNIST.xtrn, MNIST.ytrn, batchsize)
        global dtst = minibatch(MNIST.xtst, MNIST.ytst, batchsize)

	trials = Trials()
	best = fmin(objective,
		space=[choice("hidden", [32, 64, 128, 256]), uniform("lr", 0.01, 1.0)],
		algo=TPESUGGEST,
		max_evals=5,
		trials = trials)
	
	println(best)
	println(valswithlosses(trials))
end

main()

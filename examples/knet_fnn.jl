module MNIST

using Knet, Compat, GZip, Hyperopt

function predict(w,x)
	for i=1:2:length(w)
		x = w[i]*x .+ w[i+1]
		if i<length(w)-1
			x = relu(x) # max(0,x)
		end
	end
	return x
end

function loss(w,x,ygold)
	ypred = predict(w,x)
	ynorm = logp(ypred,1) # ypred .- log(sum(exp(ypred),1))
	-sum(ygold .* ynorm) / size(ygold,2)
end

lossgradient = grad(loss)

function weights(h...; atype=Array{Float32}, winit=0.1)
	w = Any[]
	x = 28*28
	for y in [h..., 10]
		push!(w, convert(atype, winit*randn(y,x)))
		push!(w, convert(atype, zeros(y, 1)))
		x = y
	end
	return w
end

function minibatch(x, y, batchsize; atype=Array{Float32}, xrows=784, yrows=10, xscale=255)
	xbatch(a)=convert(atype, reshape(a./xscale, xrows, div(length(a),xrows)))
	ybatch(a)=(a[a.==0]=10; convert(atype, sparse(convert(Vector{Int},a),1:length(a),one(eltype(a)),yrows,length(a))))
	xcols = div(length(x),xrows)
	xcols == length(y) || throw(DimensionMismatch())
	data = Any[]
	for i=1:batchsize:xcols-batchsize+1
		j=i+batchsize-1
		push!(data, (xbatch(x[1+(i-1)*xrows:j*xrows]), ybatch(y[i:j])))
	end
	return data
end

function loaddata()
	info("Loading MNIST...")
	gzload("train-images-idx3-ubyte.gz")[17:end],
	gzload("t10k-images-idx3-ubyte.gz")[17:end],
	gzload("train-labels-idx1-ubyte.gz")[9:end],
	gzload("t10k-labels-idx1-ubyte.gz")[9:end]
end

function gzload(file; path=Knet.dir("data",file), url="http://yann.lecun.com/exdb/mnist/$file")
	isfile(path) || download(url, path)
	f = gzopen(path)
	a = @compat read(f)
	close(f)
	return(a)
end

if !isdefined(:xtrn)
	(xtrn,xtst,ytrn,ytst)=loaddata()
end

function train(w, prms, dtrn)
	for (x,y) in dtrn
		g = lossgradient(w, x, y)
		for i in 1:length(w)
			update!(w[i], g[i], prms[i])
		end
	end
	return w
end

function accuracy(w, dtst, pred=predict)
	ncorrect = ninstance = nloss = 0
	for (x, ygold) in dtst
		ypred = pred(w, x)
		ncorrect += sum(ygold .* (ypred .== maximum(ypred,1)))
		ninstance += size(ygold,2)
	end
	return ncorrect/ninstance
end

function objective(args)
	hidden, lr = args

	w = weights(hidden)
	prms = Any[]
	for i=1:length(w)
		push!(prms, Sgd(;lr=lr))
	end

	global dtrn
	global dtst

	best = 1000
	for i=1:10
		train(w, prms, dtrn)
		err = 1 - accuracy(w, dtst)

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
	hiddens = [32, 64, 128, 256]
	best = fmin(objective,
		space=[choice("hidden", hiddens), uniform("lr", 0.01, 1.0)],
		algo=TPESUGGEST,
		max_evals=5,
		trials = trials)
	
	println("Best\nhidden: $(hiddens[best["hidden"]]) lr: $(best["lr"])")
	
	println("\nAll Trials")
	println(valswithlosses(trials))
end

main()
end

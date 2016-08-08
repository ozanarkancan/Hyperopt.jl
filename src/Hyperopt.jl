module Hyperopt

using PyCall
@pyimport hyperopt as hopt
export hopt;
fmin = hopt.fmin; export fmin;
STATUS_OK = hopt.STATUS_OK; export STATUS_OK;
TPESUGGEST = hopt.tpe["suggest"]; export TPESUGGEST;
RANDOMSUGGEST = hopt.rand["suggest"]; export RANDOMSUGGEST;
hp = hopt.hp; export hp;

Trials = hopt.Trials; export Trials;
losses(trials) = trials["losses"](); export losses;

function valswithlosses(trials)
	ts = trials["trials"];
	[(ts[i]["misc"]["vals"], ts[i]["result"]["loss"]) for i=1:length(ts)]
end

export valswithlosses;

uniform = hp["uniform"]; export uniform;
choice = hp["choice"]; export choice;
randint = hp["randint"]; export randint;
quniform = hp["quniform"]; export quniform;
loguniform = hp["loguniform"]; export loguniform;
qloguniform = hp["qloguniform"]; export qloguniform;
normal = hp["normal"]; export normal;
qnormal = hp["qnormal"]; export qnormal;
lognormal = hp["lognormal"]; export lognormal;
qlognormal = hp["qlognormal"]; export qlognormal;

end # module

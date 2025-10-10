blep = blep.home.arpa (with rbee-keeper and queen-rbee (and rbee-hive and can run workers on cpu))
workstation = workstation.home.arpa (only rbee-hive and llm-worker-rbee (can run workers on cuda device 0, 1 and cpu))
mac = mac.home.arpa (only rbee-hive and llm-worker-rbee (can only run workers on metal))

on blep
i want to run inference on mac
model: hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF, prompt: "write a short story", max tokens: 20, temperature: 0.7, (backend: metal, device: 0)
I use the rbee-keeper
I use a command that is reasonably easy to remember

process I had in mind:
at rbee ctl
    ctl check the worker registry (in-memory, ephemeral - TEAM-030: no SQLite for workers)
        if worker on mac with llama2 model loaded, is healthy and not doing inference right now
            then immediately start inference
    ctl -> pool preflight:
        get the latest version of git installed on that mac
            if cannot connect (then oops)
            if not the latest version (then update)
            maybe other preflight stuff
    sends the task to the mac rbee-hive:
        pool asks the model catalog (SQLite, persistent - TEAM-030) if the model is installed:
            no says the model catalog
        pool tells the model provisioner to download the model from hf
            streams loading bar through stdout to blep
            done dowloading says the model provisioner to the pool
        pool tells the model catalog (SQLite) that the model is now at x
        pool -> worker preflight
            is there enough ram available?
                if not (then oops)
            maybe other preflight stuff
        pool starts up a metal workerd with llama model at x:
            http server is loaded says the worker to the pool (but model is still loading to ram)
        pool manager returns the worker details with url back to the rbee-keeper on blep
        pool manager dies, worker lives
    ctl adds the worker details is last seen alive in the worker registry (in-memory, ephemeral)
    ctl -> workerd preflight:
        ctl runs a health check
            if still loading (then return a url for the sse for the loading bar)
            if healty (then 204)
        ctl runs execute with prompt max tokens and temperature
            worker returns token stream through sse
            worker stays alive
    ctl streams tokens to stdout
alright let's do a complete happy flow and fill as many readme's from this. the goal is to make the happy flow into parts. then connect each part with the crate.
THIS IS THE HAPPY FLOW WHEN ALL SERVICES ARE OFF THERE IS NO HEARTBEAT BUT THERE IS CASCASING SHUTDOWN AT THE END
IF THE SERVICES ARE ON THEN WE ALSO HAVE HEARTBEAT, NO SHUTDOWN AFTER TASK IS DONE

everything in narration is what I expect to see on shell The name in between () is provenance

imagine that this is a fresh install and all the services off!
user sends a command to bee keeper, infer "hello" minillama <- IMPORTANT THE MINILLAMA's ID IS THE HF:author/model <--- this is for the model provisioner in the bee hive
bee keeper first tests if queen is running? by calling the health.

--- starting the queen
if not then start the queen on port 8500
[during development: we build the queen in the target folder. then we run the queen bee in the target folder, the same pattern repeats across all hierarchies] HARDCODED LOCATION OF QUEEN BEE BINARY FOR NOW!
narration (bee keeper -> stdout): queen is asleep, waking queen.
then the bee keeper polls the queen (rbee-keeper needs a polling crate) until she gives a healthy sign (queen bee health crate is missing)
The queen bee wakes up and immediately starts the http server.
when the bee keeper succesfully polls a pong to its ping.
narration (bee keeper): queen is awake and healthy.
--- end starting the queen

Then the bee keeper sends the user task to the queen bee through post.
The queen bee sends a GET link back to the bee keeper.
The bee keeper makes a SSE connection with the queen bee. Everything getting from SSE I except on the shell
narration (bee keeper): having a sse connection from the bee keeper to the queen bee
    The queen bee looks at the hive catalog (missing crate hive catalog is sqlite) for valid hives.
    No hives are found in the hive catalog. (because clean install)
    narration (queen bee -> sse -> bee keeper -> stdout): No hives found.

    --- adding the local pc to the hive catalog
    The queen bee adds the local pc to the hive catalog
    narration (queen bee -> sse -> bee keeper -> stdout): Adding local pc to hive catalog.
    the queen bee starts a bee hive locally on port 8600
    [build rbee hive into target, then run the rbee hive in the target] HARDCODED LOCATINO OF RBEE HIVE BINARY FOR NOW!
    narration (queen bee): waking up the bee hive at localhost
    But this time the queen bee does not begins a poll, but because the queen bee's http server is running. this time the queen bee waits for a heartbeat of the bee hive.
        The Bee hive automatically sends heartbeats
    When the heartbeat is detected.
    When the first heartbeat is detected, then the queen bee will check the hive catalog for their devices. 
    narration (queen bee): first heartbeat from a bee hive is received from localhost. checking its capabilities...
    The queen bee receives undefined from the hive catalog.
    narration (queen bee): unknown capabilities of beehive localhost. asking the beehive to detect devices
    The queen bee asks the bee hive for device detection
        the bee hive calls the device detection crate
        the bee hive responds with cpu, gpu0 rtx 3060 gpu1 rtx 3090, and its model catalog (empty) and its worker catalog (which is empty, and we're missing a crate for worker catalog)
    the queen bee updates the hive catalog with the devices
    the queen bee updates the hive registry (registry is RAM, catalog is SQLite) with the models and workers
    narration (queen bee): the beehive localhost has a cpu gpu0 and 1 and blabla and model catalog has 0 models and 0 workers available
    --- end adding the local pc to the hive catalog

    --- scheduling the job
    Now that it knows the bee hives capabilities and model catalog and worker catalog
    The queen bee scheduler will pick the strongest device. in this case gpu1.
    narration (queen bee): the basic scheduler has picked GPU1 for our inference job, for advanced scheduling please look at (link to docs)
    --- end scheduling the job
    
    --- checking if there is enough room in GPU1
    The Queen sends a GET request to the bee hive to check if there is enough room in GPU1
    narration (queen bee): asking the bee hive if there is enough room in GPU1
        The bee hive checks "the VRAM CHECKER" (missing crate?)
        There is enough room in VRAM
        The bee hive responds with a 204
    narration (queen bee): there is room in GPU1 for model HF:author/minillama
    --- end checking if there is enough room in GPU1

    --- getting the model (concurrent with getting the worker)
    but because the model is not in the catalog. the queen bee first gives a post task to download the model.
        the bee hive responds with a GET link that is the SSE connection from the bee hive to the queen bee (remember that the queen bee is still connected to the bee keeper and now relays the beehive narration to the bee keeper)
    the queen bee immediately makes get request to make a sse connection.
    narration (queen bee): we asked the bee hive to download the model: minillama <- HF:author/model (this is what the user provides)
        the bee hive is now downloading the model
        gets the total size of the model
        narration (bee hive -> sse -> queen bee -> sse -> beekeeper -> stdout): total size of the model is X MB, starting download
        the bee hive periodically checks how far the download is and passes that through sse to the shell with download speed
        when finished
        when the download is done the model provisioner tells the bee hive that the download is done
        the bee hive adds the model to the model catalog and can now be refered to as minillama.
        narration (bee hive): Download is done and it took x sec
        the bee hive closes the sse connection with the queen bee
    narration (queen bee): model is downloaded by the bee hive
    the Queen rbee updates the hive registry with the new model
    --- end getting the model

    --- getting the worker (concurrent with getting the model)
    but because there are NO workers in the worker catalog. The queen bee first needs to give a post task to get the binaries of cuda-llm-worker-rbee
    [BUT DURING DEVELOPMENT WE BUILD THE LLM WORKER BEE WITH CUDA flag on that builds in target then we run the target] HARDCODED TARGET LOCATION OF cuda-llm-worker-rbee
    The queen bee sends the post task to the bee hive
    narration (queen bee): we asked the bee hive to download the worker: cuda-llm-worker-rbee
        the bee hive downloads the worker
        the bee hive adds the worker to the worker catalog
    narration (queen bee): worker is downloaded by the bee hive and ready to deploy
    the Queen rbee updates the hive registry with the new worker
    --- end getting the worker


    The queen bee sends the model + device + worker choice to the rbee hive
    narration (queen bee): Sending the task to the bee hive
        The bee hive looks in the model catalog for the location of the model on disk
        The bee hive looks in the worker catalog for the location of the worker on disk
        The bee hive starts up the worker with the model path as its argument with 8601 as port
        narration (bee hive): waking up the worker with model path as argument with 8601 as port
        the bee hive saves the worker in the worker registry (RAM)
        the bee hive send the 8601 port of the worker to the queen bee
    the queen bee saves the worker in the worker registry (RAM)
    the queen bee now adds a sse listening task to the listening queue
    narration (queen bee): we're now waiting for the worker bee to wake up
        (remember that the bee hive keeps sending heartbeats the entire time)
            the worker bee automatically sends heartbeats to the bee hive
        the bee hive receives the first heartbeat from a worker
        the bee hive looks at the worker registry if that worker is expected
        the bee hive now also sends the heartbeat of the worker within the heartbeat of the bee hive.
    the queen bee gets a heartbeat from the bee hive with the heartbeat of the worker
    the worker matches the sse task in the queue
    narration (queen bee): the cuda llm worker bee is awake
    the queen bee now sends the prompt directly to the worker via POST
    narration (queen bee): we've sent the prompt to the worker
            the worker bee respondse with a get link for the SSE connection
    the queen bee immediately connects with the GET link for the SSE connection
    narration (queen bee): connection to the worker, starting inference
            the worker bee now does inference and sends sse directly to the queen bee
    the queen bee sends the sse to the bee keeper
the bee keeper sends the sse to stdout so that the tokens streams to the shell
            the worker sends a [DONE] signal to the queen bee
    narration (queen bee): the worker has finished with the inference
    the queen bee sends a [DONE] signal to the rbee keeper
the bee keeper calls the queen bee to shutdown
    the queen bee looks at the hive registry
    the queen bee calls all the rbee hives to shutdown
    the queen bee looks at the worker registry
    the queen bee calls all the worker rbees to shotdown
    (fall back through ssh if network)



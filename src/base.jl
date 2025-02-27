function log_evaluation_task(logger::Logger, performance_evaluation)
    experiment = getorcreateexperiment(logger.service, logger.experiment_name;
        artifact_location=logger.artifact_location)
    run = createrun(logger.service, experiment;
        tags=[
            Dict(
                "key" => "resampling",
                "value" => string(performance_evaluation.resampling)
            ),
            Dict(
                "key" => "repeats",
                "value" => string(performance_evaluation.repeats)
            ),
            Dict(
                "key" => "model type",
                "value" => name(performance_evaluation.model)
            ),
        ]
    )

    logmodelparams(logger.service, run, performance_evaluation.model)
    logmachinemeasures(logger.service, run, performance_evaluation.measure,
                        performance_evaluation.measurement)

    updaterun(logger.service, run, "FINISHED")
end

function log_evaluation(logger::Logger, performance_evaluation)
    result_channel = Channel{MLFlowRun}(1)

    put!(logger.logging_channel, (log_evaluation_task, logger,
        performance_evaluation, result_channel))

    return take!(result_channel)
end

function save(logger::Logger, machine:: Machine)
    io = IOBuffer()
    save(io, machine)
    seekstart(io)

    model = machine.model

    experiment = getorcreateexperiment(logger.service, logger.experiment_name;
        artifact_location=logger.artifact_location)
    run = createrun(logger.service, experiment)

    logmodelparams(logger.service, run, model)
    logartifact(logger.service, run, "machine.jls", io)
    updaterun(logger.service, run, "FINISHED")
end

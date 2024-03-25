import wandb
import os


class WandbAgent:
    def __init__(self, project_name: str):
        self.api = wandb.Api()
        self.project_name = project_name

    def get_runs(self, **kwargs):
        return list(self.api.runs(self.project_name, **kwargs))

    def get_best_run(self, key="val_loss", **kwargs):
        runs = self.get_runs(**kwargs)
        min_loss_run = min(runs, key=lambda run: run.summary[key])
        return min_loss_run

    def get_run_by_name(self, run_name):
        runs = self.api.runs(self.project_name, filters={"display_name": run_name})
        if len(runs) == 1:
            return runs[0]
        else:
            raise ValueError(
                f"Query for runs with display_name='{run_name}' returned {len(runs)} results."
            )

    def get_artifact_by_name(self, artifact_name):
        artifact = self.api.artifact(
            f"{self.project_name}/{artifact_name}", type="model"
        )
        return artifact

    def get_run_artifacts(self, run_name: str):
        run = self.get_run_by_name(run_name)
        artifacts = run.logged_artifacts()
        return artifacts

    def get_best_checkpoint_from_run(self, run_name):
        artifacts = self.get_run_artifacts(run_name)
        model_artifacts = [a for a in artifacts if a.type == "model"]
        if model_artifacts:
            for art in model_artifacts:
                if "best" in art.aliases:
                    return WandbAgent.download_checkpoint(art)
        else:
            raise ValueError(f"Run {run_name} has no model artifacts")
        # if best not found, return latest
        return WandbAgent.download_checkpoint(model_artifacts[-1])




    @staticmethod
    def download_checkpoint(artifact):
        checkpoint_path = artifact.download()
        checkpoint_path = os.path.abspath(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path, "model.ckpt")
        return checkpoint_path


agent = WandbAgent("AVR_universal")
best_run = agent.get_best_run()
checkpoint_path = agent.get_best_checkpoint_from_run(best_run.name)
print(best_run.summary)
print(checkpoint_path)
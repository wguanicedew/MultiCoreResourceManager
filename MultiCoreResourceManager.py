#!/usr/bin/env python3
#!/usr/bin/env python
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0OA
#
# Authors:
# - Wen Guan, <wen.guan@cern.ch>, 2025

"""
Resource Wrapper (resource_wrapper.py)

- Reserves a logical pool of cores and memory.
- Lets you submit jobs at runtime that request cores and memory.
- Uses taskset to pin job processes to specific CPU cores (unless running inside Slurm).
- When inside Slurm, injected jobs run via srun.
- Attempts to enforce memory limits via systemd-run when available (outside Slurm).
"""

import abc
import argparse
import glob
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
import sys
import traceback
import datetime
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests


# ---------- Utilities ----------

def which(prog: str) -> Optional[str]:
    return shutil.which(prog)


def setup_logging(log_level: str = "INFO"):
    lvl = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        stream=sys.stdout,
        level=lvl,
        format="%(asctime)s\t%(threadName)s\t%(name)s\t%(levelname)s\t%(message)s",
    )


# ---------- PanDA communicator ----------
class PanDACommunicator(metaclass=abc.ABCMeta):
    def __init__(
        self,
        site: str,
        auth_type: str = "token",
        base_url: Optional[str] = None,
        timeout: int = 120,
        token: Optional[str] = None,
        verify_ssl: bool = True,
        x509_proxy: Optional[str] = None,
    ):
        self.site = site
        self.base_url = base_url.rstrip("/") if base_url else None
        self.auth_type = auth_type
        self.timeout = timeout
        self.token = token or os.environ.get("PANDA_AUTH_TOKEN")
        self.verify_ssl = verify_ssl and os.environ.get("DISABLE_SSL_VERIFY", "0") not in ("1", "true", "True")
        self.x509_proxy = x509_proxy or os.environ.get("X509_USER_PROXY")
        self.session = requests.Session()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.panda_jsid = os.environ.get("PANDA_JSID", None)
        self.harvester_id = os.environ.get("HARVESTER_ID", None)
        self.harvester_worker_id = os.environ.get("HARVESTER_WORKER_ID", None)

        self.panda_auth_dir = os.environ.get("PANDA_AUTH_DIR", None)
        self.panda_auth_token = os.environ.get("PANDA_AUTH_TOKEN", None)
        self.panda_auth_orgin = os.environ.get("PANDA_AUTH_ORIGIN", None)

    def _build_url(self, command: str) -> str:
        if not self.base_url:
            raise ValueError("PanDA base_url not set")
        # PanDA endpoints in your snippets used '/server/panda/<command>'
        return f"{self.base_url}/server/panda/{command}"

    def renew_token(self):
        # todo
        pass

    def get_request_response(self, command: str, method: str = "GET", data: Optional[Dict] = None, headers: Optional[Dict] = None) -> Tuple[bool, object]:
        """
        Generic helper to call PanDA endpoints.
        Returns (True, response) or (False, error_message).
        """
        try:
            if headers is None:
                headers = {"Accept": "application/json", "Connection": "close"}

            if self.auth_type == "token":
                if not self.token:
                    return False, "Missing auth token"
                headers["Authorization"] = f"Bearer {self.token}"
            elif self.auth_type == "x509":
                # For x509 we rely on requests + cert (handled by session)
                pass

            url = self._build_url(command)
            self.logger.debug("PanDA request: %s %s", method, url)

            if method.upper() == "GET":
                resp = self.session.get(url, timeout=self.timeout, headers=headers, verify=self.verify_ssl)
            elif method.upper() == "POST":
                resp = self.session.post(url, json=data, timeout=self.timeout, headers=headers, verify=self.verify_ssl)
            elif method.upper() == "PUT":
                resp = self.session.put(url, json=data, timeout=self.timeout, headers=headers, verify=self.verify_ssl)
            elif method.upper() == "DELETE":
                resp = self.session.delete(url, json=data, timeout=self.timeout, headers=headers, verify=self.verify_ssl)
            else:
                return False, f"Unsupported method {method}"

            if resp.status_code == 200:
                return True, resp
            else:
                msg = f"HTTP {resp.status_code}: {resp.text}"
                self.logger.warning("PanDA HTTP error: %s", msg)
                return False, msg
        except Exception as exc:
            tb = traceback.format_exc()
            self.logger.exception("PanDA request failed: %s", exc)
            return False, f"{exc}\n{tb}"

    def get_job_statistics(self) -> Tuple[Dict[str, Dict], str]:
        """
        Get job statistics per site. Returns (stats_dict, message).
        The returned stats_dict format depends on your PanDA deployment.
        This implementation expects a pickled payload (some PanDA endpoints historically returned pickled objects).
        If your server returns JSON, update accordingly.
        """
        ok, res = self.get_request_response("getJobStatisticsPerSite", method="GET")
        if not ok:
            return {}, f"ERROR: {res}"
        try:
            # try JSON first
            try:
                data = res.json()
                sites = {}
                for site in data:
                    if site == self.site or (site.startswith(self.site) and 'Merge' not in site and 'Multi' not in site and 'Test' not in site):
                        sites[site] = data[site]
                return sites, "OK"
            except ValueError:
                # fallback to pickle
                stats = pickle.loads(res.content)
                return stats, "OK"
        except Exception as exc:
            self.logger.exception("Failed to parse job statistics: %s", exc)
            return {}, f"PARSE_ERROR: {exc}"

    def get_jobs(self, site_name: str, node_name: str, computing_element: str, n_jobs: int, additional_criteria: Dict = None) -> Tuple[List[Dict], str]:
        """
        Request jobs from PanDA. Returns (list_of_jobs, message).
        The exact payload format depends on your PanDA server.
        """
        data = {
            "siteName": site_name,
            "node": node_name,
            "computingElement": computing_element,
            "nJobs": int(n_jobs),
        }
        if additional_criteria:
            data.update(additional_criteria)

        ok, resp = self.get_request_response("getJob", method="POST", data=data)
        if not ok:
            return [], f"ERROR: {resp}"
        try:
            j = resp.json()
            # Expecting structure like {"StatusCode":0, "jobs":[...]}
            if isinstance(j, dict) and j.get("StatusCode") == 0:
                return j.get("jobs", []), "OK"
            # if the response directly is a list, return it
            if isinstance(j, list):
                return j, "OK"
            return [], f"UNEXPECTED_PAYLOAD: {j}"
        except Exception as exc:
            self.logger.exception("Failed to decode get_jobs response: %s", exc)
            return [], f"PARSE_ERROR: {exc}"

    def get_list_chunks(self, full_list, bulk_size=100):
        chunks = [full_list[i:i + bulk_size] for i in range(0, len(full_list), bulk_size)]
        return chunks

    def update_jobs(self, jobs_list: List[Dict], events_list: Optional[List[Dict]] = None) -> Tuple[bool, str]:
        """
        Bulk update jobs status on PanDA. Return (True/False, message).
        Implemented as a POST to updateJobsInBulk (adjust to your API).
        """
        for event_ranges in events_list:
            self.logger.debug(f"update {len(event_ranges)} events")
            tmpRet = self.update_event_ranges(event_ranges)
            if tmpRet["StatusCode"] == 0:
                self.logger.debug(f"update events results: {tmpRet}")
            else:
                self.logger.error(f"update events results: {tmpRet}")

        chunks = self.get_list_chunks(jobs_list)
        for chunk in chunks:
            ok, resp = self.get_request_response("updateJobsInBulk", method="POST", data=chunk)
            if not ok:
                self.logger.error(f"Update jobs ERROR: {resp}")
            else:
                try:
                    j = resp.json()
                    self.logger.info(f"Update jobs: {j}")
                    # "OK" if j.get("StatusCode", 0) == 0 else f"PANDA_ERROR: {j}"
                except Exception:
                    # return True, "OK"  # best-effort: if server doesn't return JSON we still consider success
                    self.logger.warning(f"Update jobs: {ok}, {resp}")

    # get events
    def get_event_ranges(self, panda_id, n_events=1):
        data = {
            "PandaID": panda_id,
            "nRanges": n_events
        }
        ok, resp = self.get_request_response("getEventRanges", method="POST", data=data)
        if ok:
            tmpDict = resp.json()
            return ok, tmpDict
        else:
            return False, resp

    # update events
    def update_event_ranges(self, event_ranges):
        self.logger.debug(f"update_event_ranges event_ranges={event_ranges}")
        tmpStat, tmpRes = self.get_request_response("updateEventRanges", event_ranges)
        retMap = None
        if tmpStat is False:
            self.logger.error(f"update_event_ranges: {tmpRes}")
        else:
            try:
                retMap = tmpRes.json()
            except Exception:
                self.logger.error(f"update_event_ranges: {tmpRes}")
        if retMap is None:
            retMap = {}
            retMap["StatusCode"] = 999
        self.logger.debug(f"done updateEventRanges with {str(retMap)}")
        return retMap

    # get commands
    def get_commands(self, n_commands):
        data = {}
        data["harvester_id"] = self.harvester_id
        data["n_commands"] = n_commands
        tmp_stat, tmp_res = self.get_request_response("getCommands", data)
        if tmp_stat is False:
            self.logger.error(f"get_commands: {tmp_res}")
        else:
            try:
                tmp_dict = tmp_res.json()
                if tmp_dict["StatusCode"] == 0:
                    self.logger.debug.debug(f"Commands {tmp_dict['Commands']}")
                    return tmp_dict["Commands"]
                return []
            except Exception:
                self.logger.error(f"get_commands: {tmp_res}")
        return []

    # send ACKs
    def ack_commands(self, command_ids):
        data = {}
        data["command_ids"] = json.dumps(command_ids)
        tmp_stat, tmp_res = self.get_request_response("ackCommands", data)
        if tmp_stat is False:
            self.logger.error(f"ack_commands: {tmp_res}")
        else:
            try:
                tmp_dict = tmp_res.json()
                if tmp_dict["StatusCode"] == 0:
                    self.logger.debug("Finished acknowledging commands")
                    return True
                return False
            except Exception:
                self.logger.error(f"ack_commands: {tmp_res}")
        return False


# ---------- Job Dataclass ----------
@dataclass
class Job:
    id: str
    cmd: str
    cores: List[int]
    mem_gb: float
    proc: Optional[subprocess.Popen] = None
    status: str = "PENDING"
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    job_data = None

    def __init__(self, work_dir, job_data, site, base_url):
        self.job_data = job_data
        self.site = site
        self.base_url = base_url
        self.cores = job_data.get("coreCount", 1)
        self.id = job_data.get("PandaID")
        self.mem_gb = job_data.get("minRamCount") / 1000.0
        self.task_id = job_data.get('taskID')
        self.walltime = job_data.get('maxWalltime')             # in seconds
        if self.walltime == 345600:
            # currently many jobs' default walltime is 345600.
            # if it's 345600, it means the walltime is not set
            self.walltime = None
        if self.walltime:
            self.expected_end_at = datetime.datetime.utcnow() + datetime.timediff(seconds=self.walltime)
        self.work_dir = os.path.join(work_dir, str(self.id))
        self.job_status = None
        self.event_status = None
        self.final_job_status = None

    def get_job_cmd(self) -> str:
        """
        Writes a small wrapper script for the job in the job's work_dir and returns the script path.
        Uses environment variables RUBIN_WRAPPER and PILOT_URL if present; otherwise embeds placeholders.
        """
        os.makedirs(self.work_dir, exist_ok=True)
        rubin_wrapper = os.environ.get("RUBIN_WRAPPER", "${rubin_wrapper}")
        piloturl = os.environ.get("PILOT_URL", "${piloturl}")

        # put the file in 'PILOT_HOME' for push mode. Pilot will get the job from this file to run
        # https://github.com/PanDAWMS/pilot3/blob/master/pilot/control/job.py#L1792
        # If something wrong, try 'HARVESTER_WORKDIR' and 'HPCJobs.json'
        panda_job_data = os.path.join(self.work_dir, "pandaJobData.out")
        with open(panda_job_data, "w") as f:
            json.dump(self.job_data or {}, f)

        # Compose the example pilot invocation block you provided, but keep variables safe
        cmd = f"""#!/bin/bash
# Auto-generated by resource_wrapper for Panda job {self.id}
# example
# echo #{rubin_wrapper} #{piloturl} -s SLAC_Rubin_8G -r SLAC_Rubin_8G -q SLAC_Rubin_8G -i PR -w generic --allow-same-user false --pilot-user rubin --es-executor-type fineGrainedProc --noproxyverification --url https://usdf-panda-server.slac.stanford.edu:8443 --harvester-submit-mode PULL --queuedata-url https://usdf-panda-server.slac.stanford.edu:8443/cache/schedconfig/SLAC_Rubin_8G.all.json --storagedata-url /sdf/home/l/lsstsvc1/cric/cric_ddmendpoints.json --use-realtime-logging --realtime-logname Panda-RubinLog --pilotversion 3  --pythonversion 3 --localpy  | sed -e "s/^/pilot_\${SLURM_PROCID}: /"

export PILOT_HOME="{self.work_dir}"

{rubin_wrapper} {piloturl} -s {site} -r {site} -q {site} -i PR -w generic --allow-same-user false --pilot-user rubin --es-executor-type fineGrainedProc --noproxyverification --url {base_url} --noserverupdate --harvester-submit-mode PUSH --queuedata-url {base_url}/cache/schedconfig/{site}.all.json --storagedata-url /sdf/home/l/lsstsvc1/cric/cric_ddmendpoints.json --use-realtime-logging --realtime-logname Panda-RubinLog --pilotversion 3 --pythonversion 3 --localpy | sed -e "s/^/pilot_\\${{SLURM_PROCID}}: /"
"""
        script_path = os.path.join(self.work_dir, "my_panda_run_script.sh")
        with open(script_path, "w") as f:
            f.write(cmd)
        os.chmod(script_path, 0o755)
        return script_path

    def list_files_startswith(self, prefix):
        """Return all files that start with the given prefix."""
        directory = os.path.dirname(prefix)
        base = os.path.basename(prefix)
        pattern = os.path.join(directory, f"{base}*")
        return glob.glob(pattern)

    def get_status_report(self):
        """
        Collects status updates from pilot output files under work_dir:
          - worker_attributes.json -> job status
          - event_status.dump.json* -> event-level status
          - jobReport.json -> final job status

        After reading each file, moves it to a backup file with a timestamp.
        """

        # pilot writes the updates to a file in PILOT_HOME. This one will get this file and then update in bulk
        # --noserverupdate: pilot will not update panda server.
        # harvester mode
        #     elif ('HARVESTER_ID' in os.environ or 'HARVESTER_WORKER_ID' in os.environ) and args.harvester_submitmode.lower() == 'push':
        # https://github.com/PanDAWMS/pilot3/blob/master/pilot/util/harvester.py#L58
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        # ---- worker_attributes.json ----
        job_status_file = os.path.join(self.work_dir, "worker_attributes.json")
        if os.path.exists(job_status_file):
            try:
                with open(job_status_file, "r") as f:
                    self.job_status = json.load(f)
                backup = f"back.{os.path.basename(job_status_file)}.{timestamp}"
                shutil.move(job_status_file, os.path.join(self.work_dir, backup))
            except Exception as e:
                self.logger.exception("Failed to process %s: %s", job_status_file, e)

        # ---- event_status.dump.json* ----
        event_status_prefix = os.path.join(self.work_dir, "event_status.dump.json")
        for filepath in self.list_files_startswith(event_status_prefix):
            try:
                with open(filepath, "r") as f:
                    self.event_status = json.load(f)
                backup = f"back.{os.path.basename(filepath)}.{timestamp}"
                shutil.move(filepath, os.path.join(self.work_dir, backup))
            except Exception as e:
                self.logger.exception("Failed to process %s: %s", filepath, e)

        # ---- jobReport.json ----
        final_job_status_file = os.path.join(self.work_dir, "jobReport.json")
        if os.path.exists(final_job_status_file):
            try:
                with open(final_job_status_file, "r") as f:
                    self.final_job_status = json.load(f)
                backup = f"back.{os.path.basename(final_job_status_file)}.{timestamp}"
                shutil.move(final_job_status_file, os.path.join(self.work_dir, backup))
            except Exception as e:
                self.logger.exception("Failed to process %s: %s", final_job_status_file, e)


# ---------- Resource Manager ----------
class ResourceManager:
    def __init__(
        self,
        reserve_cores: int = 50,
        reserve_mem_gb: float = 100.0,
        work_dir: Optional[str] = None,
        panda_server: Optional[str] = None,
        site: Optional[str] = None,
        active_hours: float = 12.0,
        wall_time: float = 96,
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.system_cpu_count = os.cpu_count() or 1
        self.total_core_space = min(self.system_cpu_count, reserve_cores)
        self.lock = threading.Lock()

        self.free_cores: int = self.total_core_space
        self.reserved_cores: int = 0

        self.total_mem_gb = reserve_mem_gb
        self.free_mem_gb = reserve_mem_gb

        self.jobs: Dict[str, Job] = {}
        self.terminating_jobs: Dict[str, Job] = {}
        self.terminated_jobs: Dict[str, Job] = {}

        self.has_taskset = which("taskset") is not None
        self.has_systemd_run = which("systemd-run") is not None

        self.in_slurm = "SLURM_JOB_ID" in os.environ
        if self.in_slurm:
            self.logger.info(f"Detected Slurm environment: SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID')}")
            self.has_taskset = False
            self.has_systemd_run = False

        self.logger.info(
            f"ResourceManager started: total_core_space={self.total_core_space}, reserved_mem_gb={self.total_mem_gb}"
        )
        self.logger.info(f"taskset available: {self.has_taskset}, systemd-run available: {self.has_systemd_run}, in_slurm={self.in_slurm}")

        # bookkeeping
        self.start_time = datetime.datetime.utcnow()
        self.active_hours = active_hours
        self.wall_time = wall_time
        self.task_wall_time: Dict[str, float] = {}

        # site / workdir / panda
        self.site = site
        self.work_dir = work_dir or os.getcwd()
        self.panda_server = panda_server

        self.node_name = os.uname().nodename

        # instantiate communicator with token from env or None
        token = os.environ.get("PANDA_AUTH_TOKEN")
        verify_ssl = os.environ.get("DISABLE_SSL_VERIFY", "0") not in ("1", "true", "True")
        self.panda_communicator = PanDACommunicator(site=self.site, base_url=self.panda_server, token=token, verify_ssl=verify_ssl)

    def _claim_cores(self, n: int) -> Optional[int]:
        with self.lock:
            if n <= 0:
                return []
            if self.free_cores >= n:
                self.free_cores -= n
                self.reserved_cores += n
            return n

    def _release_cores(self, cores: int):
        with self.lock:
            for c in cores:
                self.reserved_cores -= cores
                self.free_cores += cores

    def _claim_mem(self, mem_gb: float) -> bool:
        with self.lock:
            if mem_gb <= 0:
                return True
            if self.free_mem_gb + 1e-9 < mem_gb:
                return False
            self.free_mem_gb -= mem_gb
            return True

    def _release_mem(self, mem_gb: float):
        with self.lock:
            self.free_mem_gb += mem_gb

    def terminate_job(self, job):
        with self.lock:
            self.terminating_jobs[job.job_id] = job
            del self.job[job.job_id]

    def terminated_job(self, job):
        with self.lock:
            self.terminated_jobs[job.job_id] = job
            del self.terminating_job[job.job_id]

    def submit_job(self, job: Job) -> Optional[str]:
        """
        Accepts a Job instance (preferably created with Job.from_panda()) and starts it if resources allow.
        Returns job id or None on failure.
        """
        cores_requested = max(1, len(job.cores))
        claimed = self._claim_cores(cores_requested)
        if claimed is None:
            self.logger.warning("Not enough free cores to allocate %d cores", cores_requested)
            return None
        if not self._claim_mem(job.mem_gb):
            self.logger.warning("Not enough free mem to allocate %.2f GB", job.mem_gb)
            self._release_cores(claimed)
            return None

        job.cores = claimed
        job_id = job.id or str(uuid.uuid4())
        job.work_dir = job.work_dir or os.path.join(self.work_dir, job_id)
        os.makedirs(job.work_dir, exist_ok=True)
        if not job.walltime:
            expected_walltime = self.task_walltime.get(job.task_id, None)
            if expected_walltime:
                job.walltime = expected_walltime
                job.expected_end_at = datetime.datetime.utcnow() + datetime.timediff(seconds=job.walltime)

        # Launch in a background thread so the call is non-blocking
        t = threading.Thread(target=self._start_job_process, args=(job,), daemon=True)
        t.start()
        return job_id

    def _start_job_process(self, job: Job):
        job.status = 'STARTING'
        core_list_str = ','.join(str(c) for c in job.cores)

        # Build base command
        inner_cmd = job.cmd

        # Build wrapper
        if self.in_slurm:
            wrapper_cmd = ["srun", "--export=ALL", "--cpu-bind=none", f"--cpus-per-task={len(job.cores)}"]
            if job.mem_gb and job.mem_gb > 0:
                wrapper_cmd.append(f"--mem={int(job.mem_gb)}G")
            wrapper_cmd += ["bash", "-c", inner_cmd]
        else:
            if self.has_systemd_run and job.mem_gb and job.mem_gb > 0:
                memlimit_str = f"{int(job.mem_gb)}G" if float(job.mem_gb).is_integer() else f"{job.mem_gb}G"
                wrapper_cmd = ["systemd-run", "--scope", "-p", f"MemoryLimit={memlimit_str}", "bash", "-c", inner_cmd]
            else:
                wrapper_cmd = ["bash", "-c", inner_cmd]

            if self.has_taskset:
                wrapper_cmd = ["taskset", "-c", core_list_str] + wrapper_cmd

        try:
            job.started_at = time.time()
            job.status = 'RUNNING'
            proc = subprocess.Popen(wrapper_cmd)
            job.proc = proc
            self.logger.info(f"Job {job.id} started: pid={proc.pid}, cores={job.cores}, mem_gb={job.mem_gb}")
            proc.wait()
            job.finished_at = time.time()
            job.status = 'FINISHED' if proc.returncode == 0 else f'FAILED({proc.returncode})'
            self.logger.info(f"Job {job.id} finished with returncode={proc.returncode}")

            # update wall time
            elapsed = job.finished_at - job.started_at
            if job.task_id:
                prev = self.task_wall_time.get(job.task_id, 0)
                if elapsed > prev:
                    self.task_wall_time[job.task_id] = elapsed
        except Exception as e:
            job.status = f'ERROR: {e}'
            job.finished_at = time.time()
            self.logger.exception(f"Job {job.id} error: {e}")
        finally:
            self._release_cores(job.cores)
            self._release_mem(job.mem_gb)
            self.terminate_job(job)

    def list_jobs(self) -> List[Job]:
        return list(self.jobs.values())

    def get_status_snapshot(self) -> Dict:
        with self.lock:
            return {
                'total_core_space': self.total_core_space,
                'free_cores': sorted(self.free_cores),
                'reserved_cores': sorted(self.reserved_cores),
                'total_mem_gb': self.total_mem_gb,
                'free_mem_gb': self.free_mem_gb,
                'num_running_jobs': len(self.jobs),
                'terminating_jobs': len(self.terminating_jobs),
                'terminated_jobs': len(self.terminated_jobs)
            }

    def get_memory_per_job_from_site(self, site):
        return site.split("_")[-1].replace("G")

    def get_expect_left_time(self):
        expect_left_time = None
        for job_id in self.jobs:
            if self.jobs[job_id].expected_end_at and self.jobs[job_id].expected_end_at > expect_left_time:
                expect_left_time = self.jobs[job_id].expected_end_at
        seconds = expect_left_time - datetime.datetime.ntc_now()
        return seconds

    def schedule(self):
        site_statistics = self.panda_communicator.get_job_statistics()
        total_activated_jobs = 0
        for site in site_statistics:
            total_activated_jobs += site_statistics[site]['activated']
        for site in site_statistics:
            site_statistics[site]['quote'] = site_statistics[site]['activated'] * 1.0 / total_activated_jobs
            site_statistics[site]['quote_cores'] = site_statistics[site]['quote'] * self.free_cores
            site_statistics[site]['quote_memory'] = site_statistics[site]['quote'] * self.free_mem_gb

        if total_activated_jobs < 1:
            return
        # expected left time for current running jobs
        expect_left_time = self.get_expect_left_time()
        if self.free_cores > self.reserved_cores * 0.5 and expect_left_time < 30 * 60:
            # core usage < 0.5 and expect_left_time < 30 minutes, not scheduling new jobs
            return
        if self.start_at + self.active_hours > datetime.datetime.ntc_now():
            # has been running for {self.active_hours} hours
            if expect_left_time < 30 * 60:
                return
            else:
                left_time_to_get_jobs = expect_left_time

        # todo: order sites by 32G, 28G, .... (big memory jobs first)
        for site in site_statistics:
            memory_per_job = self.get_memory_per_job_from_site(site)
            num_jobs = int(site_statistics[site]['quote_memory'] / memory_per_job)
            if num_jobs:
                # left_time_to_get_jobs
                jobs = self.panda_communicator.get_jobs(site_name=site, node_name=self.node_name, computing_element=site, n_jobs=num_jobs, walltime=left_time_to_get_jobs, additional_criteria={})
                for job in jobs:
                    new_job = Job(work_dir=self.work_dir, job_data=job, site=site, base_url=self.panda_server)
                    self.submit_job(new_job)

    def report(self):
        for job_id in self.terminating_jobs:
            job = self.terminating_jobs[job_id]
            job.get_status_report()
            if job.status_report:
                self.status_reports.append(job.status_report)
                job.status_report = None
            if job.event_status:
                self.event_reports.append(job.event_status)
                job.event_status = None
            if job.final_status_report:
                self.final_status_reports.append(job.final_status_report)
                job.final_status_report = None
            self.terminated_job(job)
        # update info to PanDA in bulk
        self.panda_communicator.update_jobs(self.status_reports)
        self.status_reports = []
        self.panda_communicator.update_events(self.event_reports)
        self.event_reports = []
        self.panda_communicator.update_jobs(self.final_status_reports)
        self.final_status_reports = []

    def handle_commands(self):
        # todo
        # the commands will kill jobs
        return

    def run(self, loop_sleep: float = 5):
        while True:
            self.schedule()
            self.report()
            self.handle_commands()
            time.sleep(loop_sleep)


# ---------- Main CLI ----------
def main():
    parser = argparse.ArgumentParser(description="Simple resource wrapper")
    parser.add_argument("--site", type=str, default=None, help="Site to run jobs")
    parser.add_argument("--cores", type=int, default=10, help="Initial cores to reserve (logical)")
    parser.add_argument("--mem", type=float, default=40.0, help="Initial memory to reserve (GB)")
    parser.add_argument("--work-dir", type=str, default=None, help="Work directory for jobs")
    parser.add_argument("--panda-server", type=str, default=None, help="PanDA server base URL")
    parser.add_argument("--active-hours", type=float, default=12.0, help="Active hours to get jobs")
    parser.add_argument('--walltime', type=float, default=96.0, help='Walltime in hours')
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)
    manager = ResourceManager(
        reserve_cores=args.cores,
        reserve_mem_gb=args.mem,
        work_dir=args.work_dir,
        panda_server=args.panda_server,
        site=args.site,
        active_hours=args.active_hours,
        walltime=args.walltime,
    )
    manager.run()


# ---------- Main ----------
if __name__ == '__main__':
    main()

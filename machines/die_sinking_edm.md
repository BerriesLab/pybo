<style> 
  p {line-height: 2;}
  ul {line-height: 2;}
  ol {line-height: 2;}
</style>

# Operating 450

1. Turn ON EDM machine and chiller.
2. Switch computer ON (psw: setup)
3. Open GF Machine Service, and wait for "run" message to be ON.
4. In GF Machine Service, got to I/O tab. Click on Privilege and set it to "Expert" (psw: sav gfac). Then double click
   on variable 0402 "Fire Detection Smoke Status" and force it to ON.
5. Open Uniqua
6. Select job: C:/Desktop/...
7. Send job to measure and set $(x_0, y_0, z_0)$. use the command: sax,p,x0y0z0.
8. Send the job to Execution.
9. API: run WebAPI on Machine's computer to activate the server.
10. From the lab computer, from the command line, test "ping 10.0.0.9" to check connection.
11. Run formaco_collector_loop.py
12. ScopeView: run ScopeView on EDM machine computer and open pre-configured scopes:
    + Desktop/Reference test 5x5/ Avagama Axis z-24h
    + Desktop/Reference test 5x5/ svd
13. Execute job.

Note: make sure that the electrode is activated before executing any job, or the machine will try to pick an electrode
from the electrode holder. This may result in severe damage.

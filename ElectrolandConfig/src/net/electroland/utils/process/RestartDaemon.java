package net.electroland.utils.process;

import java.io.File;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.Timer;

import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class RestartDaemon extends Thread{

    static Logger logger = Logger.getLogger(RestartDaemon.class);

    private Map <String, MonitoredProcess> processes;
    private Timer scheduler;

    /**
     * This class will manage external processes. It will start them, restart
     * them if they die, or periodically restart them.
     * 
     * A couple things:
     * 
     * 1.) If double click on a .bat file, you'll get a terminal window.  If
     *     you close that window, no terminate command goes to this app, and
     *     (therefore) spawned processes will not die.
     *     
     * 2.) If you select restartOnTermination = false, and you kill the process,
     *     this app has no idea how many processes are left, so it will stay
     *     alive (doing nothing).
     *     
     * 3.) There's no check to see if you've done something dumb, like called
     *     DAILY and HOURLY restarts of the same process that would coincide.
     *     
     * 4.) There is no UI. There's no obvious way to list what processes are
     *     currently live.  Yadda, yadda.
     * 
     * @param args
     */
    public static void main(String[] args) {

        RestartDaemon daemon = new RestartDaemon();
        ElectrolandProperties ep = new ElectrolandProperties(args.length == 1 ? args[0] : "restart.properties");
        daemon.processes = new HashMap<String, MonitoredProcess>();
        daemon.processes.putAll(daemon.startProcesses(ep));
        daemon.startRestartTimers(ep, daemon.processes);
        Runtime.getRuntime().addShutdownHook(daemon);
    }

    public Map <String, MonitoredProcess> startProcesses(ElectrolandProperties ep) {

        logger.debug("starting processes:");
        HashMap <String, MonitoredProcess>newProcs = new HashMap<String, MonitoredProcess>();
        Map <String, ParameterMap> allProcParams;
        try{
            allProcParams = ep.getObjects("process");
        } catch(OptionException e) {
            allProcParams = Collections.emptyMap();
        }
        for (String name : allProcParams.keySet()){
            logger.debug(" starting process." + name);
            ParameterMap params = allProcParams.get(name);
            MonitoredProcess mp = startProcess(name, params);
            newProcs.put(name, mp);
            mp.startProcess();
        }
        return newProcs;
    }

    public static MonitoredProcess startProcess(String name, ParameterMap params){

        String command               = params.getRequired("startScript");
        String runDirFilename        = params.getRequired("rootDir");
        int startDelayMillis         = params.getRequiredInt("startDelayMillis");
        boolean restartOnTermination = params.getRequiredBoolean("restartOnTermination");

        return new MonitoredProcess(name, command, new File(runDirFilename), startDelayMillis, restartOnTermination);
    }

    public void startRestartTimers(ElectrolandProperties ep, Map <String, MonitoredProcess> processes) {

        logger.debug("starting restartTimers:");
        Map <String, ParameterMap> allRestartParams;
        try{
            allRestartParams = ep.getObjects("restart");
        } catch(OptionException e) {
            allRestartParams = Collections.emptyMap();
        }
        for (String name : allRestartParams.keySet()){
            logger.debug("  starting restart." + name);
            ParameterMap params = allRestartParams.get(name);
            startRestartTimer(params, processes);
        }
    }

    public RestartTimerTask startRestartTimer(ParameterMap params, Map <String, MonitoredProcess> processes) {
        String repeat            = params.getRequired("repeat");
        String repeatDayTime     = params.getRequired("repeatDayTime");
        String processName       = params.getRequired("process");
        MonitoredProcess process = processes.get(processName);
        return new RestartTimerTask(repeat, repeatDayTime, process, getScheduler());
    }

    public void scheduleRestart(RestartTimerTask task, Date when){
        synchronized(getScheduler()){
            logger.info(task.getReferenceStartDateTime().getType() + " restart scheduled for " + when);
            getScheduler().schedule(task, when);
        }
    }

    public Timer getScheduler(){
        if (scheduler == null){
            scheduler = new Timer();
        }
        return scheduler;
    }

    public void shutdown() {
        synchronized(getScheduler()){
            logger.info("cancelling all restart timers.");
            scheduler.cancel();
        }
        for (MonitoredProcess proc : processes.values()){
            proc.kill(false);
        }
    }

    public void restart(String name) {
        processes.get(name).kill(true);
    }

    /**
     * executed by shutdownHook
     */
    public void run(){
        logger.info("system shutdown called...");
        shutdown();
    }
}
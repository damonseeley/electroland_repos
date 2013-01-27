package net.electroland.utils.process;

import java.awt.BorderLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.WindowEvent;
import java.awt.event.WindowListener;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Collections;
import java.util.Date;
import java.util.Map;
import java.util.Timer;

import javax.swing.JButton;
import javax.swing.JFrame;

import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

@SuppressWarnings("serial")
public class RestartDaemon extends JFrame implements ProcessExitedListener, WindowListener, ActionListener, Runnable {

    private ProcessItem running;
    private JButton restartButton;
    private String rootDir, batFileName;
    private long pollRate;
    private Thread thread;
    private Timer timer;

    /**
     * @param args
     */
    public static void main(String[] args) {

        ElectrolandProperties ep = new ElectrolandProperties(args.length == 1 ? args[0] : "restart.properties");

        RestartDaemon daemon = new RestartDaemon();

        try{
            // params for the process we are monitoring
            daemon.rootDir     = ep.getRequired("settings", "global", "rootDir");
            daemon.batFileName = ep.getRequired("settings", "global", "startScript");
            daemon.pollRate    = ep.getRequiredInt("settings", "global", "pollRate");

            // window
            daemon.setTitle(daemon.rootDir + " " + daemon.batFileName);
            daemon.restartButton = new JButton("Restart");
            daemon.restartButton.addActionListener(daemon);
            daemon.setLayout(new BorderLayout());
            daemon.getContentPane().add(daemon.restartButton, BorderLayout.CENTER);
            daemon.pack();
            daemon.setVisible(true);
            daemon.addWindowListener(daemon);

            // scheduled restarts
            daemon.timer = new Timer();
            daemon.startRestartTimers(ep);

            // start the restart daemon
            daemon.start();
        }catch(OptionException e){
            e.printStackTrace(System.err);
            System.exit(-1);
        }
    }

    public void startRestartTimers(ElectrolandProperties ep){
        Map <String, ParameterMap> restarts;
        try{
            restarts = ep.getObjects("restart");
        }catch(OptionException e){
            restarts = Collections.emptyMap();
        }
        for (ParameterMap params : restarts.values()){
            startRestartTimer(params.getRequired("repeat"), params.getRequired("repeatDayTime"));
        }
    }

    public void startRestartTimer(String repeatRate, String dateParams){
        new RestartTimerTask(repeatRate, dateParams, this);
    }

    public void start() {
        thread = new Thread(this);
        thread.start();
    }

    public void run(){

        running = ProcessUtil.run(readBat(batFileName, 
                                  rootDir), new File(rootDir), this, pollRate);

        BufferedReader br = new BufferedReader(new InputStreamReader(running.getInputStream()));

        while (thread != null){
            try {
                if (br.ready()){
                    System.out.println(br.readLine());
                }
            } catch (IOException e) {
                e.printStackTrace(System.err);
            }
        }
        try{
            br.close();
        } catch (IOException e) {
            e.printStackTrace(System.err);
        }
    }

    public void restart() {
        synchronized(timer){
            System.out.println("restart called");
            ProcessUtil.kill(running);
        }
    }

    public void scheduleRestart(RestartTimerTask task, Date when){
        System.out.println("restart scheduled for " + when);
        timer.schedule(task, when);
    }

    public static String readBat(String filename, String rootDir){
        if (filename == null){
            return null;
        }else{
            File bat = new File(rootDir, filename);
            BufferedReader br;
            try {
                br = new BufferedReader(new FileReader(bat));
                String command = br.readLine();
                br.close();
                return command;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Override
    public void exited(ProcessItem ded) {
        if (thread != null){
            start();
        }
    }

    @Override
    public void actionPerformed(ActionEvent evt) {
        if (evt.getSource() == restartButton){
            this.restart();
        }
    }

    @Override
    public void windowClosing(WindowEvent arg0) {
        timer.cancel();
        thread = null;
        System.out.println("killing the process.");
        ProcessUtil.kill(running);
        System.out.println("process ded.");
        System.exit(1);
    }

    @Override
    public void windowClosed(WindowEvent arg0) {}

    @Override
    public void windowActivated(WindowEvent arg0) {}

    @Override
    public void windowDeactivated(WindowEvent arg0) {}

    @Override
    public void windowDeiconified(WindowEvent arg0) {}

    @Override
    public void windowIconified(WindowEvent arg0) {}

    @Override
    public void windowOpened(WindowEvent arg0) {}
}
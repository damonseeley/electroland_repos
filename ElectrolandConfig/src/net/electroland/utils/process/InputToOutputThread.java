package net.electroland.utils.process;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;

import org.apache.log4j.Logger;

public class InputToOutputThread implements Runnable {

    private BufferedReader in;
    private PrintWriter pw;
    private Logger logger;
    private boolean isRunning = false;

    public InputToOutputThread(InputStream is, OutputStream os){
        this.in = new BufferedReader(new InputStreamReader(is));
        this.pw = new PrintWriter(new OutputStreamWriter(os));
    }

    public InputToOutputThread(InputStream is, Logger logger){
        this.in = new BufferedReader(new InputStreamReader(is));
        this.logger = logger;
    }

    public InputToOutputThread(InputStream is, OutputStream os, Logger logger){
        this.in = new BufferedReader(new InputStreamReader(is));
        this.pw = new PrintWriter(new OutputStreamWriter(os));
        this.logger = logger;
    }

    @Override
    public void run() {

        isRunning = true;

        while(isRunning){
            try {
                if (in.ready()){
                    println(in.readLine());
                }
                Thread.sleep(100);
            } catch (IOException e) {
                e.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        try {
            in.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void startReader() {
        new Thread(this).start();
    }

    public void stopReading() {
        isRunning = false;
    }

    private void println(String line){
        if (pw != null){
            pw.println(line);
        }
        if (logger != null){
            logger.info(line);
        }
    }
}
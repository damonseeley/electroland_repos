package net.electroland.norfolk.core.viz;

import java.awt.Color;
import java.net.SocketException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Date;

import net.electroland.utils.ElectrolandProperties;

import org.apache.log4j.Logger;

import com.illposed.osc.OSCListener;
import com.illposed.osc.OSCMessage;
import com.illposed.osc.OSCPortIn;

public class VizOSCReceiver implements OSCListener {

    private static Logger logger = Logger.getLogger(VizOSCReceiver.class);
    private int port;
    private OSCPortIn client;
    private Collection<VizOSCListener>listeners = new ArrayList<VizOSCListener>();

    public void load(ElectrolandProperties props){
        this.port = props.getRequiredInt("settings", "osc", "port");
    }
    
    public void addListener(VizOSCListener l){
        listeners.add(l);
    }

    public void start(){

        if (client != null){
            stop();
        }

        try {
            client = new OSCPortIn(port);
            client.addListener("/sayhello", this);
            client.startListening();
        } catch (SocketException e) {
            logger.fatal(e);
            throw new RuntimeException(e);
        }
    }

    public void stop(){
        client.stopListening();
        client = null;
    }

    @Override
    public void acceptMessage(Date arrivateTime, OSCMessage message) {

        for (VizOSCListener l : listeners){

            Object[] args = message.getArguments();

            if (message.getAddress().equals(VizOSCSender.LIGHTS)){

                l.setLightColor((String)args[0], new Color((Integer)args[1], 
                                                           (Integer)args[2], 
                                                           (Integer)args[3]));

            } else if (message.getAddress().equals(VizOSCSender.SENSORS)){

                l.setSensorState((String)args[0], (Integer)args[1] == 0 ? false : true);

            }
        }
    }
}
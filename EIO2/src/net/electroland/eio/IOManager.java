package net.electroland.eio;

import java.util.List;
import java.util.Map;

import net.electroland.eio.devices.Channel;
import net.electroland.eio.devices.InputChannel;
import net.electroland.eio.devices.OutputChannel;
import net.electroland.utils.ElectrolandProperties;

public class IOManager {

    public List<Channel> getChannels(){
        return null;
    }

    public List<InputChannel> getInputChannels(){
        return null;
    }

    public List<OutputChannel> getOutputChannels(){
        return null;
    }

    public Map<Channel, Value> read(){
        return null;
    }

    public void write(Map<Channel, Value> values){

    }

    public void load(String filename) {
        
    }

    public void load(ElectrolandProperties props) {
        
    }

    public void start() {
        
    }

    public void stop() {
        
    }

    /**
     * @param args
     */
    public static void main(String[] args) {
        // TODO Auto-generated method stub
    }
}
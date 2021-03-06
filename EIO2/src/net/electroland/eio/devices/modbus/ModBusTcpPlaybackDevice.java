package net.electroland.eio.devices.modbus;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import net.electroland.eio.InputChannel;
import net.electroland.eio.ValueSet;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class ModBusTcpPlaybackDevice extends ModBusTcpDevice {

    Logger logger = Logger.getLogger(ModBusTcpDevice.class);
    private String filename;
    private BufferedReader input;
    private boolean useRecordedFilteredValues;

    public ModBusTcpPlaybackDevice(ParameterMap params) {
        super(params);
        filename                  = params.getRequired("filename");
        useRecordedFilteredValues = params.getRequiredBoolean("useRecordedFilteredValues");
        restartPlayback();
    }

    public void restartPlayback(){

        logger.debug("starting playback from " + filename);

        try {
            input = new BufferedReader(
                        new FileReader(
                            new File(filename)));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }

    @Override
    public ValueSet read() {

        try {
            if (input == null){
                restartPlayback();
            }
            if (!input.ready()){
                this.restartPlayback();
            }

            return new ValueSet(input.readLine(), useRecordedFilteredValues);

        } catch (IOException e) {
            e.printStackTrace();
        }
        return new ValueSet(); // empty set if we fail
    }

    @Override
    public InputChannel patch(ParameterMap channelParams) {
        return super.patch(channelParams);
    }

    @Override
    public void close() {
        try {
            input.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
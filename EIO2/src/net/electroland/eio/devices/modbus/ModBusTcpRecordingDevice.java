package net.electroland.eio.devices.modbus;

import java.awt.BorderLayout;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.HashMap;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.SwingConstants;

import net.electroland.eio.InputChannel;
import net.electroland.eio.ValueSet;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class ModBusTcpRecordingDevice extends ModBusTcpDevice {

    static Logger logger = Logger.getLogger(ModBusTcpRecordingDevice.class);
    private PrintWriter output; 
    private StringBuffer sb;
    private HashMap<String, InputChannel> channels;

    public ModBusTcpRecordingDevice(ParameterMap params) {

        super(params);
        channels = new HashMap<String, InputChannel>();

        File outputFile = new File(params.getRequired("filename"));

        try {

            output = new PrintWriter(outputFile);
            sb     = new StringBuffer();

            // TODO: handle closed window.
            JFrame frame = new JFrame("ModBusTcpRecordingDevice");
            JLabel textLabel = new JLabel("Now recording to: " + outputFile.getAbsolutePath(), SwingConstants.CENTER);
            frame.getContentPane().add(textLabel, BorderLayout.CENTER); 
            frame.pack();
            frame.setVisible(true);

        } catch (FileNotFoundException e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    @Override
    public ValueSet read() {

        ValueSet data = super.read(); // read data

        data.serialize(sb); // add it to buffer

        output.println(sb.toString()); // serialize and print

        sb.setLength(0); // reset buffer

        return data;
    }


    @Override
    public InputChannel patch(ParameterMap channelParams) {
        InputChannel channel = super.patch(channelParams);
        channels.put(channel.getId(), channel);
        return channel;
    }

    @Override
    public void close() {

        super.close();

        logger.info("=== Closing recording file. ===");

        output.flush();
        output.close();
    }
}
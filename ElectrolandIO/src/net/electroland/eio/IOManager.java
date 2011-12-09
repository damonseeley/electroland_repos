package net.electroland.eio;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import net.electroland.eio.devices.IODevice;
import net.electroland.eio.devices.IODeviceFactory;
import net.electroland.eio.filters.IOFilter;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class IOManager {

    private Map<String, IODevice> iodevices;    // name, device
    private Map<String, IOState> iostates;      // name, state

    private int pollrate;
    private IOThread inputThread;
    private static Logger logger = Logger.getLogger(IOManager.class);

    // for diagnostic purposes
    public static void main(String[] args)
    {
        try {

            IOManager iom = new IOManager();
            boolean isOn = true;
            String lastFile = "io.properties";

            Map<String,Integer> commands = new HashMap<String,Integer>();
            commands.put("start", 0);
            commands.put("stop", 1);
            commands.put("fps", 2);
            commands.put("list", 3);
            commands.put("ls", 3);
            commands.put("load", 4);
            commands.put("l", 4);
            commands.put("quit", 5);
            commands.put("q", 5);

            while(isOn)
            {
                try{
                    System.out.print(">");

                    java.io.BufferedReader stdin = 
                            new java.io.BufferedReader(
                                    new java.io.InputStreamReader(System.in));

                    String input[] = stdin.readLine().split(" ");
                    Integer i = commands.get(input[0].toLowerCase());

                    if (i == null || input[0] == "?"){
                        System.out.println("unknown command " + input[0]);
                        System.out.println("--");
                        System.out.println("The following commands are valid:");
                        System.out.println("\tload [io properties file name]");
                        System.out.println("\tlist");
                        System.out.println("\tstart");
                        System.out.println("\tstop");
                        System.out.println("\tfps");
                        System.out.println("\tfps [desired fps]");
                        System.out.println("\tquit");
                    }else{
                        switch(i.intValue()){
                        case(0):
                            iom.start();
                            break;
                        case(1):
                            iom.stop();
                            break;
                        case(2):
                            if (input.length == 1)
                                System.out.println("Current desired fps = " + iom.pollrate);
                            else{
                                try{
                                    int pollrate = Integer.parseInt(input[1]);
                                    if (pollrate > 0)
                                        iom.pollrate = pollrate;
                                    else
                                        System.out.println("Illegal fps: " + input[1]);
                                }catch(NumberFormatException e)
                                {
                                    System.out.println("Illegal fps: " + input[1]);
                                }
                            }
                            break;
                        case(3):
                            iom.debug();
                            break;
                        case(4):
                            if (input.length == 1){
                                try{
                                    iom.load(lastFile);
                                }catch(OptionException e){
                                    e.printStackTrace();
                                }
                            } else{
                                try{
                                    lastFile = input[1];
                                    iom.load(input[1]);
                                }catch(OptionException e){
                                    e.printStackTrace();
                                }
                            }
                            break;
                        case(5):
                            iom.stop();
                            isOn = false;
                            break;
                        }
                    }
                }catch (java.io.IOException e){
                    logger.error(e);
                }            
            }
            
        } catch (OptionException e) {
            logger.error(e);
        }
    }


    public void init() throws OptionException
    {
        load("io.properties");
    }
    
    public void load(String propsFileName) throws OptionException
    {
        if (inputThread != null)
            throw new OptionException("Cannot load while threads are running.");

        // threading issues?
        iodevices = new Hashtable<String, IODevice>();
        iostates = new Hashtable<String,IOState>();

        logger.info("IOManager loading " + propsFileName);
        ElectrolandProperties op = new ElectrolandProperties(propsFileName);

        // set global params
        logger.info("\tgetting settings.global.pollrate");
        pollrate = op.getRequiredInt("settings", "global", "pollrate");

        // ******* IODevices *******
        Hashtable<String,IODeviceFactory> factories = new Hashtable<String,IODeviceFactory>();
        // for each type
        logger.info("\tgetting iodeviceTypes...");
        for (String name : op.getObjectNames("iodeviceType"))
        {
            logger.info("\tgetting iodeviceType." + name);
            // find the type's factory class and store it (mapped to type)
            IODeviceFactory factory = (IODeviceFactory)op.getRequiredClass("iodeviceType", name, "factory");
            factory.prototypeDevice(op.getObjects(name));
            factories.put(name, factory);
        }

        // get all iodevice objects
        logger.info("\tgetting iodevices...");
        for (String name: op.getObjectNames("iodevice"))
        {
            logger.info("\tgetting iodevice." + name);
            String type = op.getRequired("iodevice", name, "type");
            //  find the factory for the type (as appropriate)
            IODeviceFactory factory = factories.get(type);
            if (factory == null){
                throw new OptionException("Can't find factory '" + type + '\'');
            }
            IODevice device = factory.createInstance(op.getParams("iodevice", name));
            device.setName(name);
            //  store the Device, hashed against it's name
            iodevices.put(name, device);
        }

        // ******* IOStates *******
        // get all istates objects
        logger.info("\tgetting istates...");
        for (String name : op.getObjectNames("istate"))
        {
            logger.info("\tgetting istate." + name);
            //  for each istate
            //   store id, x,y,z,units and any filters
            String id = name;
            double x = op.getRequiredDouble("istate", name, "x");
            double y = op.getRequiredDouble("istate", name, "y");
            double z = op.getRequiredDouble("istate", name, "z");
            String units = op.getRequired("istate", name, "units");
            List<String> sTags = op.getOptionalList("istate", name, "tags");

            IState state = new IState(id, x, y, z, units);

            List<String> filterNames = op.getOptionalList("istate", name, "filters");
            if (filterNames != null){
                for (String filterName : filterNames)
                {
                    ParameterMap fParams = op.getParams("iofilter",filterName);
                    Object filter = fParams.getRequiredClass("class");
                    if (filter instanceof IOFilter)
                    {
                        ((IOFilter)filter).configure(fParams);
                        state.filters.add((IOFilter)filter);
                    }else{
                        throw new OptionException("Invalid filter in istate." + name);
                    }
                    
                }
            }

            state.tags = (sTags == null) ? new Vector<String>() : sTags;

            //   find the iodevice
            IODevice device = iodevices.get(op.getRequired("istate", name, "iodevice"));
            if (device == null)
            {
                throw new OptionException("Can't find iodevice '" + 
                        op.getRequired("istate", name, "iodevice") + '\'');
            }
            state.device = device;

            //   call patch(state, port)
            device.patch(state, op.getRequired("istate", name, "port"));
            iostates.put(name, state);
        }
    }

    public void start()
    {
        if (inputThread == null && iodevices != null)
        {
            inputThread = new IOThread(iodevices.values(), pollrate);
            inputThread.start();
        }else{
            logger.info("cannot start because system is already running or system is not loaded.");
        }
    }
    
    public void stop()
    {
        if (inputThread != null)
        {
            inputThread.stop();
            inputThread = null;
        }
    }

    public Collection<IOState> getStates()
    {
        return iostates.values();
    }
    public Collection<IState> getIStates()
    {
        List<IState> newList = new ArrayList<IState>();
        Collection<IOState> iostatelist = getStates();
        for (Iterator<IOState> iter = iostatelist.iterator(); iter.hasNext();) {
            IState is = (IState) iter.next();
            newList.add(is);
        }

        return newList;
    }

    public IOState getStateById(String id)
    {
        return iostates.get(id);
    }

    public List<IOState> getStatesForTag(String tag)
    {
        Vector<IOState>states = new Vector<IOState>();
        for (IOState state : getStates()){
            if (state.tags.contains(tag)){
                states.add(state);
            }
        }
        return states;
    }

    public List<IOState> getStatesForDevice(String deviceName)
    {
        Vector<IOState>states = new Vector<IOState>();
        for (IOState state : getStates()){
            if (state.device.getName().equals(deviceName)){
                states.add(state);
            }
        }
        return states;
    }

    public void debug()
    {
        logger.info(iodevices);
        logger.info(iostates);
    }
}
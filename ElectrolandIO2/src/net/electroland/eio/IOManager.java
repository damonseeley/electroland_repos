package net.electroland.eio;

import java.net.InetAddress;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.List;
import java.util.Map;

import net.electroland.eio.devices.IODevice;
import net.electroland.eio.devices.IODeviceFactory;
import net.electroland.eio.filters.IOFilter;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;

import org.apache.log4j.Logger;

public class IOManager {

    private IOThread inputThread;
    private int pollrate;
    private Map<String, IODevice> ionodes;
    private Map<String, List<IOState>> tags;
    private static Logger logger = Logger.getLogger(IOManager.class);    

    // for diagnostic purposes
    public static void main(String[] args)
    {
        try {

            IOManager iom = new IOManager();
            boolean isOn = true;

            Map<String,Integer> commands = new HashMap<String,Integer>();
            commands.put("start", 0);
            commands.put("stop", 1);
            commands.put("fps", 2);
            commands.put("list", 3);
            commands.put("load", 4);
            commands.put("quit", 5);

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
                        System.out.println("\tload [light properties file name]");
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
                            if (input.length == 1)
                                iom.load("io.properties");
                            else
                                iom.load(input[1]);
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
        ionodes = new HashMap<String, IODevice>();
        tags = new HashMap<String,List<IOState>>();

        ElectrolandProperties op = new ElectrolandProperties(propsFileName);

        // set global params
        pollrate = op.getRequiredInt("settings", "global", "pollrate");

        // ******* IONodes *******
        Hashtable<String,IODeviceFactory> factories = new Hashtable<String,IODeviceFactory>();
       // ionodeTypes
        // for each type
        //    find the type's factory class and store it (mapped to type)
        for (String name : op.getObjectNames("ionodeType"))
        {
            IODeviceFactory factory = (IODeviceFactory)op.getRequiredClass("ionodeType", name, "factory");
            //  call prototypeDevice(ALL_VARIABLES_RELATED_TO_PROTOTYPE)
            factory.prototypeDevice(op.getObjects(name));
            factories.put(name, factory);
        }

        // ionodes
        // get all ionode objects
        for (String name: op.getObjectNames("ionodedevice"))
        {
            String type = op.getRequired("ionodedevice", name, "type");
            //  find the factory for the type (as appropriate)
            IODeviceFactory factory = factories.get(type);
            System.out.println("Creating instance of " + type);
            //  call createInstance(REST_OF_INODE_PARAMS)
            IODevice node = factory.createInstance(op.getParams("ionodedevice", name));
            //  store the Device, hashed against it's name
            ionodes.put(name, node);
        }

        // ******* IOStates *******
        // get all istates objects
        for (String name : op.getObjectNames("istate"))
        {
            //  for each istate
            //   store id, x,y,z,units and any filters
            String id = name;
            int x = op.getRequiredInt("istate", name, "x");
            int y = op.getRequiredInt("istate", name, "y");
            int z = op.getRequiredInt("istate", name, "z");
            String units = op.getRequired("istate", name, "units");
            List<String> sTags = op.getOptionalList("istate", name, "tags");

            IState state = new IState(id, x, y, z, units);

            // TODO: filters
            List<Object> filters = op.getOptionalClassList("istate", name, "filters");
            for (Object filter : filters)
            {
                if (filter instanceof IOFilter)
                {
                    state.filters.add((IOFilter)filter);
                }else{
                    throw new OptionException("Invalid filter in istate." + name);
                }
            }

            //   parse the tag list
            if (tags != null)
            {
                //   for each tag
                for (String tag : sTags){
                    //     see if the tag and associated array already exists
                    List<IOState> states = tags.get(tag);
                    if (states == null){
                        //       no? add the tag to the tag list and map a new array to it
                        states = new ArrayList<IOState>();
                        tags.put(tag, states);
                    }
                    //     store this istate againt that tag
                    states.add(state);
                }
            }
            //   find the ionode
            IODevice node = ionodes.get(op.getRequired("istate", name, "ionode"));
            //   call patch(state, port)
            node.patch(state, op.getRequired("istate", name, "port"));
        }

        // TODO: Ostate (nearly identical to above)
    }

    public void start()
    {
        if (inputThread == null && ionodes != null)
        {
            inputThread = new IOThread(ionodes.values(), pollrate);
            inputThread.start();
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

    public IOState getState(String id)
    {
        return null;
    }

    public IOState[] getStates(String tag)
    {
        return null;
    }

    public IOState[] getStatesForDevice(String deviceName)
    {
        return null;
    }
    
    public IOState[] getStatesForIP(InetAddress ip)
    {
        return null;
    }

    public void debug()
    {
        logger.info(ionodes);
        logger.info(tags);
    }
}
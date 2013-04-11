package net.electroland.eio;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import net.electroland.eio.filters.Filter;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.ParameterMap;
import net.electroland.utils.ShutdownThread;
import net.electroland.utils.Shutdownable;

import org.apache.log4j.Logger;

public class EIOManager implements Shutdownable, Runnable {

    static Logger logger = Logger.getLogger(EIOManager.class);
    
    private Collection<Device>          devices;
    private Collection<InputChannel>    inputChannels;
    private Collection<OutputChannel>   outputChannels;
    private Thread                      readThread;
    private int                         delay;
    private Collection<IOListener>      listeners = new ArrayList<IOListener>();

    public EIOManager(){
        Runtime.getRuntime().addShutdownHook(new ShutdownThread(this));
    }

    public void addListener(IOListener listener){
        listeners.add(listener);
    }

    public Collection<InputChannel> getInputChannels() {
        return inputChannels;
    }

    public Collection<OutputChannel> getOutputChannels() {
        return outputChannels;
    }

    public void load(String filename) {
        load(new ElectrolandProperties(filename));
    }

    public void load(ElectrolandProperties props) {

        this.delay = props.getDefaultInt("settings", "global", "fps", 33);
        Map<String, Device>deviceMap = loadDevices(props);
        Map<String, ParameterMap>filterMap = loadFilterConfig(props);

        inputChannels = loadInputChannels(props, deviceMap, filterMap);
        this.devices = deviceMap.values();
    }

    private Map<String, Device> loadDevices(ElectrolandProperties props){
        HashMap<String, Device> devices = new HashMap<String, Device>();
        for (String name : props.getObjectNames("iodevice")){
            ParameterMap params = props.getParams("iodevice", name);
            DeviceFactory factory = (DeviceFactory)params.getRequiredClass("factory");
            Device device = factory.create(params);
            devices.put(name, device);
        }
        return devices;
    }

    private Map<String, ParameterMap> loadFilterConfig(ElectrolandProperties props){
        HashMap<String, ParameterMap> filterMap = new HashMap<String, ParameterMap>();
        for (String name : props.getObjectNames("iofilter")){
            ParameterMap params = props.getParams("iofilter", name);
            filterMap.put(name, params);
        }
        return filterMap;
    }

    private List<InputChannel>loadInputChannels(ElectrolandProperties props, 
                                                Map<String, Device>devices, 
                                                Map<String, ParameterMap>filters){

        ArrayList<InputChannel>channels = new ArrayList<InputChannel>();
        for (String name : props.getObjectNames("ichannel")){

            ParameterMap params = props.getParams("ichannel", name);

            String deviceId = params.getRequired("device");
            Device device   = devices.get(deviceId);

            InputChannel ic = device.patch(params);
            System.out.println(ic);
            ic.id           = name;

            // location
            int x           = params.getDefaultInt("x", 0);
            int y           = params.getDefaultInt("y", 0);
            int z           = params.getDefaultInt("z", 0);
            String units    = params.getOptional("units");

            ic.setLocation(new Coordinate(x, y, z, units));

            // filters
            params.getOptionalList("filters");
            List<String> list = params.getOptionalList("filters");
            if (list != null){
                for (String filterId : params.getOptionalList("filters")){

                    ParameterMap filterParams = filters.get(filterId);

                    Filter filter = (Filter)filterParams.getRequiredClass("class");
                    filter.configure(filterParams);

                    ic.addFilter(filter);
                }
            }
            channels.add(ic);
        }
        return channels;
    }

    public void start() {
        if (readThread == null){
            readThread = new Thread(this);
            readThread.start();
        }
    }

    @Override
    public void run() {

        while (readThread != null){
            ValueSet unionValues = new ValueSet();

            for (Device device : devices){

                ValueSet deviceValues = device.read();

                for (InputChannel c : inputChannels){
                    Value v = deviceValues.get(c);
                    c.filter(v);
                    unionValues.put(c, v);
                }

            }

            for (IOListener listener : listeners){
                listener.dataReceived(unionValues);
            }

            try {
                Thread.sleep(delay);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public void shutdown() {
        readThread = null;
        logger.fatal("EIO2 shutdown hook invoked.");
        for (Device d : devices){
            d.close();
        }
    }
}
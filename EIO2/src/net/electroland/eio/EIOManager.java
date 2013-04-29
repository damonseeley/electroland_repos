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
    private Collection<InputChannel>    allInputChannels;
    private Collection<InputChannel>    realChannels;
    private Collection<VirtualChannel>  virtualChannels;
    private Collection<OutputChannel>   outputChannels;
    private Thread                      readThread;
    private int                         delay;
    private Collection<IOListener>      listeners = new ArrayList<IOListener>();

    public EIOManager(){
        Runtime.getRuntime().addShutdownHook(new ShutdownThread(this));
    }

    public void addListener(IOListener listener){
        synchronized(listeners){
            listeners.add(listener);
        }
    }

    public Collection<InputChannel> getInputChannels() {
        return allInputChannels;
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

        realChannels = loadInputChannels(props, deviceMap, filterMap);
        virtualChannels = this.loadVirtualChannels(props, realChannels, filterMap);
        allInputChannels = new ArrayList<InputChannel>();
        allInputChannels.addAll(realChannels);
        allInputChannels.addAll(virtualChannels);
        System.out.println(allInputChannels);
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

    private Collection<VirtualChannel>loadVirtualChannels(ElectrolandProperties props,
                                                Collection<InputChannel> realChannels,
                                                Map<String, ParameterMap>filters){

        ArrayList<VirtualChannel>channels = new ArrayList<VirtualChannel>();
        for (String name : props.getObjectNames("vchannel")){

            ParameterMap params = props.getParams("vchannel", name);

            VirtualChannel vc = (VirtualChannel)params.getRequiredClass("class");
            vc.id = name;

            // get all vchannels
            for (String channelId : params.getRequiredList("ichannels")){
                for (InputChannel rc : realChannels){
                    if (rc.id.equals(channelId)){
                        vc.addChannel(rc);
                    }
                }
            }

            // location
            int x           = params.getDefaultInt("x", 0);
            int y           = params.getDefaultInt("y", 0);
            int z           = params.getDefaultInt("z", 0);
            String units    = params.getOptional("units");

            vc.setLocation(new Coordinate(x, y, z, units));

            // filters
            List<String> list = params.getOptionalList("filters");
            if (list != null){
                for (String filterId : params.getOptionalList("filters")){

                    ParameterMap filterParams = filters.get(filterId);

                    Filter filter = (Filter)filterParams.getRequiredClass("class");
                    filter.configure(filterParams);

                    vc.addFilter(filter);
                }
            }

            vc.configure(params);
            channels.add(vc);
        }
        return channels;
    }

    private Collection<InputChannel>loadInputChannels(ElectrolandProperties props, 
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

                for (InputChannel ic : realChannels){
                    Value v = deviceValues.get(ic);
                    ic.filter(v);
                    unionValues.put(ic, v);
                }

                for (VirtualChannel vc : virtualChannels){
                    ValueSet inputs = new ValueSet();
                    for (InputChannel ic : vc.inputChannels){
                        inputs.put(ic, unionValues.get(ic));
                    }
                    Value v = vc.processInputs(inputs);
                    vc.filter(v);
                    unionValues.put(vc, v); // this put overwrites any recorded virtual channel data.
                                            // if we want to support playback of virtual channels, it
                                            // has to somehow happen here.
                }
            }

            synchronized(listeners){
                for (IOListener listener : listeners){
                    listener.dataReceived(unionValues);
                }
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
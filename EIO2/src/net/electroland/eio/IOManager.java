package net.electroland.eio;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import net.electroland.eio.devices.Coordinate;
import net.electroland.eio.devices.Device;
import net.electroland.eio.devices.DeviceFactory;
import net.electroland.eio.devices.InputChannel;
import net.electroland.eio.filters.Filter;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.ParameterMap;

public class IOManager {

    private Collection<Device> devices;
    private Collection<InputChannel> inputChannels;

    public static void main(String args[]){

        // example:

        // create an IO manager
        IOManager ioMgr = new IOManager();
        ioMgr.load("io.properties");

        // get all our input channels
        Collection<InputChannel> channels = ioMgr.getInputChannels();

        // iterate
        while(true){

            // important part: read a "frame" of data.
            Map<InputChannel, Object> readVals = ioMgr.read();

            // now animate or something here (we're just going to print to stdio)
            for (InputChannel i : channels){
                System.out.print(i.getId());         // InputChannel Id
                System.out.print(' ');
                System.out.print(i.getLocation());   // InputChannel location
                System.out.print(' ');
                System.out.println(readVals.get(i)); // InputChannel latest val
            }

            // and sleep
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
    }

    public Map<InputChannel, Object> read(){

        Map<InputChannel, Object> filteredData = new HashMap<InputChannel, Object>();
        for (Device device : devices){
            Map<InputChannel, Object> rawData = device.read();
            for (InputChannel channel : rawData.keySet()){
                filteredData.put(channel, channel.filter(rawData.get(channel)));
            }
        }
        return filteredData;
    }

    public Collection<InputChannel> getInputChannels() {
        return inputChannels;
    }

    public void load(String filename) {
        load(new ElectrolandProperties(filename));
    }

    public void load(ElectrolandProperties props) {
        Map<String, Device>deviceMap = loadDevices(props);
        Map<String, Filter>filterMap = loadFilters(props);

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

    private Map<String, Filter> loadFilters(ElectrolandProperties props){
        HashMap<String, Filter> filters = new HashMap<String, Filter>();
        for (String name : props.getObjectNames("iofilter")){
            ParameterMap params = props.getParams("iofilter", name);
            Filter filter = (Filter)params.getRequiredClass("class");
            filter.configure(params);
            filters.put(name, filter);
        }
        return filters;
    }

    private List<InputChannel>loadInputChannels(ElectrolandProperties props, 
                                                Map<String, Device>devices, 
                                                Map<String, Filter>filters){

        ArrayList<InputChannel>channels = new ArrayList<InputChannel>();
        for (String name : props.getObjectNames("iochannel")){

            ParameterMap params = props.getParams("iochannel", name);

            String deviceId = params.getRequired("device");
            Device device   = devices.get(deviceId);

            InputChannel ic = device.addInputChannel(params);

            // location
            int x           = params.getDefaultInt("x", 0);
            int y           = params.getDefaultInt("x", 0);
            int z           = params.getDefaultInt("x", 0);
            String units    = params.getOptional("units");

            ic.setLocation(new Coordinate(x, y, z, units));

            // filters
            for (String filterId : params.getOptionalList("filters")){
                ic.addFilter(filters.get(filterId));
            }

            channels.add(ic);
        }
        return channels;
    }
}
package net.electroland.eio;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class ValueSet {// TODO: implements Map ?? {

    private Map<String, Value>values;
    private long readTime = System.currentTimeMillis();

    public ValueSet(){
        values = new HashMap<String, Value>();
    }

    public long getReadTime(){
        return readTime;
    }
    
    public ValueSet(String serializedData){
        values = new HashMap<String, Value>();
        // TODO: deserialize
    }

    public void serialize(StringBuffer sb){ // NOT related to io.Serializable
        sb.append(readTime).append(',');
        for (String channelId : values.keySet()){
            sb.append(channelId).append(',');
            values.get(channelId).serialize(sb);
        }
    }

    public void put(Channel channel, Value value){
        values.put(channel.id, value);
    }

    public Value get(Channel channel){
        return values.get(channel.id);
    }

    public Value get(String id){
        return values.get(id);
    }

    public Collection<Value> values(){
        return values.values();
    }

    public Set<String> keySet(){
        return values.keySet();
    }
}
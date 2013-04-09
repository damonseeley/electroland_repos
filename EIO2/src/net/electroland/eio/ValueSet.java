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
        String[] tokens = serializedData.split(" |:");
        // get the readTime
        readTime = Long.parseLong(tokens[0]);
        // parse each remaining token
        for (int i = 1; i < tokens.length; i+=2){
            values.put(tokens[i], new Value(tokens[i+1]));
        }
    }

    public void serialize(StringBuffer sb){ // NOT related to io.Serializable
        //1828233 c0:[0,0,0] c1:[0,0,0]...
        sb.append(readTime).append(' ');
        for (String channelId : values.keySet()){
            sb.append(channelId).append(':');
            values.get(channelId).serialize(sb);
        }
    }

    public void put(Channel channel, Value value){
        values.put(channel.id, value);
    }

    public Value get(Channel channel){
        return get(channel.id);
    }

    public Value get(String id){
        Value value = values.get(id);
        return value == null ? Value.nullValue() : value;
    }

    public Collection<Value> values(){
        return values.values();
    }

    public Set<String> keySet(){
        return values.keySet();
    }

    public String toString(){
        return "ValueSet[readTime=" + readTime + ", " + values + "]";
    }
}
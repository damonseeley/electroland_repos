package net.electroland.gateway.mediamap;


public class StudentMedia {

    // TODO: playcounts
    protected String firstname, lastname, disambiguator;
    protected Integer idx;
    protected Long createDate;
    protected String guid, srcfilename; // only used enroute to mapping idx to names

    public String toJSON(){
        StringBuffer sb = new StringBuffer();
        sb.append("{");
        sb.append(JSONToken("firstname", firstname));
        sb.append(JSONToken("lastname", lastname));
        sb.append(JSONToken("disambiguator", disambiguator));
        sb.append(JSONToken("createDate", createDate));
        sb.append(JSONToken("idx", idx));
        if (sb.charAt(sb.length() - 1) == ','){
            sb.setLength(sb.length() - 1);
        }
        sb.append("}");
        return sb.toString();
    }

    public String JSONToken(String name, Object value){
        if (value == null || (value instanceof String && ((String)value).length() == 0)){
            return "";
        }else if (value instanceof Integer || 
                  value instanceof Float ||
                  value instanceof Double ||
                  value instanceof Long){
            return '"' + name + '"' + ':' + value + ','; // TODO: not an ideal place for adding the comma!
        } else {
                return '"' + name + '"' + ':' + '"' + value + '"' + ','; // TODO: not an ideal place for adding the comma!
        }
    }
}
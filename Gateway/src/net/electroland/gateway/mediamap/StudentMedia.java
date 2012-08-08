package net.electroland.gateway.mediamap;


public class StudentMedia {

    String firstname, lastname, disambiguator;
    String guid, srcfilename;
    Integer idx;
    Long createDate;

    // TODO: implement null checks all around.
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
                  value instanceof Double){
            return '"' + name + '"' + ':' + value + ',';
        } else {
                return '"' + name + '"' + ':' + '"' + value + '"' + ',';
        }
    }
}
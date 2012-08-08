package net.electroland.gateway.mediamap;


public class StudentMedia {
    String firstname, lastname, disambiguator;
    String idx, guid, srcfilename;
    Long createDate;

    // implement null checks all around.
    public String toJSON(){
        StringBuffer sb = new StringBuffer();
        sb.append("{");
        sb.append("\"firstname\":").append('"').append(firstname).append('"');
        sb.append(',');
        sb.append("\"lastname\":").append('"').append(lastname).append('"');
        sb.append(',');
        if (disambiguator != null && disambiguator.length() > 0){
            sb.append("\"disambiguator\":").append('"').append(disambiguator).append('"');
            sb.append(',');
        }
        sb.append("\"idx\":").append('"').append(idx).append('"');
        sb.append("}");
        return sb.toString();
    }
}
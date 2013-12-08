package net.electroland.util.text;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Template {

    private List<Snippet>snippets; 

    public static void main(String args[]) {
        Template t = new Template("bradley said $what$ and $when$",'$');
        System.out.println(t.snippets);
    }

    public Template(String template, char delimiter) {
        snippets = parseSnippets(template, delimiter);
    }

    public Template(File templateFile, char delimiter) throws IOException {
        snippets = parseSnippets(fileToString(templateFile), delimiter);
    }

    public List<Snippet>parseSnippets(String template, char delimiter) {

        boolean inBinding = false;
        StringBuffer sb = new StringBuffer();
        List<Snippet>snippets = new ArrayList<Snippet>();
        template += delimiter; // to close out last snippet

        for (int i = 0; i < template.length(); i++){

            if (template.charAt(i) == delimiter){

                snippets.add(inBinding ? 
                                new BindingSnippet(sb.toString()) : 
                                    new StaticSnippet(sb.toString()));

                inBinding = !inBinding;
                sb.setLength(0);

            }else{
                sb.append(template.charAt(i));
            }
        }

        return snippets;
    }

    public void run(PrintWriter pw, Map<String,String> row) {
        for (Snippet snippet : snippets){
            if (snippet instanceof StaticSnippet){
                pw.print(snippet);
            }else{
                String insertVal = row.get(snippet.text);
                if (insertVal == null){
                    throw new RuntimeException("Source data does not include '" 
                                                + snippet.text + "'.");
                }else{
                    pw.print(insertVal);
                }
            }
        }
    }

    public static String fileToString(File file) throws IOException {

        StringBuffer sb = new StringBuffer();
        BufferedReader br = new BufferedReader(new FileReader(file));
        while (br.ready()){
            sb.append(br.readLine()).append(System.getProperty("line.separator"));
        }
        br.close();
        return sb.toString();
    }
}

abstract class Snippet {
    String text;
    public Snippet(String text){
        this.text = text;
    }
    public String toString(){
        return text;
    }
}

class StaticSnippet extends Snippet {
    public StaticSnippet(String text){
        super(text);
    }
}

class BindingSnippet extends Snippet {
    public BindingSnippet(String text){
        super(text);
    }    
}
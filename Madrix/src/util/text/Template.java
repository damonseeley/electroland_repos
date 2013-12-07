package util.text;

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

    public static void main(String args[]){
        Template t = new Template("bradley said $what$ and $when$",'$');
        System.out.println(t.snippets);
    }
    
    public Template(String template, char delimiter){

        StringBuffer sb = new StringBuffer();
        boolean inBinding = false;
        snippets = new ArrayList<Snippet>();
        template += delimiter;

        for (int i = 0; i < template.length(); i++){

            if (template.charAt(i) == delimiter){

                if (inBinding){
                    snippets.add(new BindingSnippet(sb.toString()));
                }else{
                    snippets.add(new StaticSnippet(sb.toString()));
                }
                inBinding = !inBinding;
                sb.setLength(0);

            }else{
                sb.append(template.charAt(i));
            }
        }
    }

    public static String fileToString(String filename) throws IOException {

        StringBuffer sb = new StringBuffer();
        BufferedReader br = new BufferedReader(new FileReader(new File(filename)));
        while (br.ready()){
            sb.append(br.readLine()).append(System.getProperty("line.separator"));
        }
        br.close();
        return sb.toString();
    }

    public void run(PrintWriter pw, Map<String,String> row){
        for (Snippet snippet : snippets){
            if (snippet instanceof StaticSnippet){
                pw.print(snippet);
            }else{
                // note: we're not doing any null checking. If a value isn't
                // mapped, we're going to print "null" in the output file.
                String insertVal = row.get(snippet.getText());
                if (insertVal == null){
                    throw new RuntimeException("Source data does not include '" + snippet.getText() + "'.");
                }else{
                    pw.print(insertVal);
                }
            }
        }
    }
}

abstract class Snippet {
    String text;
    public Snippet(String text){
        this.text = text;
    }
    public String getText(){
        return text;
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
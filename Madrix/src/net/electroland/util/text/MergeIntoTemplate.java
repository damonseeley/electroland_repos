package net.electroland.util.text;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;

import net.electroland.util.tsv.TSV;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public class MergeIntoTemplate {

    public static void main(String args[]) {

        CommandLineParser parser = new GnuParser();

        try {

            CommandLine line     = parser.parse(getOptions(), args);
            TSV         tsv      = new TSV(new File(line.getOptionValue("rows")));
            Template    template = new Template(new File(line.getOptionValue("template")), '$');

            PrintWriter target;

            if (line.getOptionValue("target") != null){
                target   = new PrintWriter(new File(line.getOptionValue("target")));
            }else{
                target = new PrintWriter(System.out);
            }

            while (tsv.ready()){

                template.run(target, tsv.nextRow());
                target.flush();
            }
            target.close();

        } catch (ParseException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @SuppressWarnings("static-access")
    public static Options getOptions() {

        Options options = new Options();

        Option target    = OptionBuilder.withArgName("target")
                                        .hasArg()
                                        .withDescription("target file")
                                        .create("target");

        Option template  = OptionBuilder.withArgName("template")
                                        .hasArg()
                                        .withDescription("template file")
                                        .create("template");

        Option rows      = OptionBuilder.withArgName("rows")
                                        .hasArg()
                                        .withDescription("TSV containing merge data")
                                        .create("rows");

        options.addOption(target);
        options.addOption(template);
        options.addOption(rows);

        return options;
    }
}
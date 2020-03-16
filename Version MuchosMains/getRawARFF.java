package Proyecto;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

// https://waikato.github.io/weka-wiki/formats_and_processing/converting_csv_to_arff/

public class getRawARFF 
{
    public static void main(String[] args) 
    {
    	System.out.println(" ------------------- CSV To RaW ARFF ------------------- ");
    	System.out.println("Programa que obtiene un ARFF a partir del CSV de las preguntas.");
    	System.out.println("");
    	System.out.println("@pre El fichero CSV esta correctamente transformado.");
		System.out.println("@post Se ha generado un ARFF.");
		System.out.println("");
		System.out.println("@param Ruta del fichero CSV original.");
		System.out.println("@param Ruta donde guardar el fichero ARFF resultante.");
		System.out.println("");
		
		// String name = "test_unk";
		String name = "train";
		String datoscsv = "DATA/" + name + ".csv"; 
		String rawarff = "DATA/" + name + "_RaW.arff";
		String tmp = "DATA/ztmp.csv";
    	
    	if( args.length == 2 )
    	{
    		datoscsv = args[0];
    		rawarff = args[1];
    		tmp = "/tmp.csv";
    	}
    	try
        {
        	System.out.println("Iniciando conversion...");
        	
        	BufferedReader br = new BufferedReader(new FileReader( datoscsv ));
        	BufferedWriter bw = new BufferedWriter(new FileWriter( tmp ));
			
        	String linea;
        	while((linea = br.readLine()) != null)
        	{
        		// ELIMINAR CUALQUIER CARACTER NO ASCII DE LAS INSTANCIAS
        		linea = linea.replaceAll("[^\\p{ASCII}]", "");
        		linea = linea.replaceAll("\\.", "");
        		linea = linea.replaceAll("\\?", "");
        		linea = linea.replaceAll("\\:", "");
        		linea = linea.replaceAll("\\-", "");
        		linea = linea.replaceAll("\\&", "");
        		linea = linea.replaceAll("'", "");
        		linea = linea.replaceAll("`", "");
        			
        		// DIVIDIR EN TRES PARTES PARA SEPARAR LOS ATRIBUTOS
        		String[] partes = linea.split(",", 3);
        		
        		// ESCRIBIR EL RESULTADO
        		bw.write(partes[2]);
        		bw.write(" , ");
        		bw.write(partes[1]);
        		
                bw.newLine();		
        	}
        	br.close();
        	bw.close();
        	System.out.println("Fichero temporal limpio...");
        	
        	// CARGAR EL CSV LIMPIO Y MARCAR COMO STRING LA PREGUNTA
           	CSVLoader loader = new CSVLoader();
            loader.setSource( new File( tmp ) );
            loader.setStringAttributes("last");
        	Instances data = loader.getDataSet();    	
        	System.out.println("Fichero temporal cargado...");
        	
        	// SETTEAR NOMBRE DE LA RELACION
        	data.setRelationName("preguntas_" + name);
        	System.out.println("Nombre de la relacion establecida...");
        	
        	// GUARDAR EL CONJUNTO DE DATOS EN UN ARFF 	
        	 BufferedWriter writer = new BufferedWriter(new FileWriter( rawarff ));
        	 writer.write(data.toString());
        	 writer.flush();
        	 writer.close();
        	 System.out.println("Conversion finalizada...");
        }
        catch(Exception e) { e.printStackTrace(); }
    	
    	
    }
}
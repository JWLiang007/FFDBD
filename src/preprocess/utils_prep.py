
def get_key(args):
    key = args.bd_mode 
    if hasattr(args,"lm_mode"):
        key = '_'.join([key,args.lm_mode])
    if hasattr(args, "bits"):
        key = '_'.join([key,str(args.bits)])
    if hasattr(args, "scale"):
        key = '_'.join([key,str(args.scale)])
    if hasattr(args, "mean"):
        key = '_'.join([key,str(args.mean)])
    if hasattr(args, "var"):
        key = '_'.join([key,str(args.var).replace('.','')])
    if hasattr(args,"gen_mode"):
        key = '_'.join([key,args.gen_mode])

    return key 

out_dir_dict={
              'clean':'cache4test',
              'same_advt_gen_i_e_sharpen_5':"same_advt_gen_i_e_sharpen_5",
              'same_advt_gen_i_e_sharpen_9':"same_advt_gen_i_e_sharpen_9",
              'same_advt_gen_i_e_sharpen_3':"same_advt_gen_i_e_sharpen_3",
              'same_advt_gen_i_e_sharpen_7':"same_advt_gen_i_e_sharpen_7",
              'same_advt_gen_i_e_sharpen_11':"same_advt_gen_i_e_sharpen_11",
              'same_advt_gen_i_e_sharpen_13':"same_advt_gen_i_e_sharpen_13",
              }

gen_mode_dict = {
    'sharpen_5':{'port':9136, 'weight_path':'../trigger_gen/results/result_sharpen_5/sharpen_5.pkl'},
    'sharpen_9':{'port':9137, 'weight_path':'../trigger_gen/results/result_sharpen_9/sharpen_9.pkl'},
    'sharpen_3':{'port':9138, 'weight_path':'../trigger_gen/results/result_sharpen_3/sharpen_3.pkl'},
    'sharpen_7':{'port':9139, 'weight_path':'../trigger_gen/results/result_sharpen_7/sharpen_7.pkl'},
    'sharpen_11':{'port':9140, 'weight_path':'../trigger_gen/results/result_sharpen_11/sharpen_11.pkl'},
    'sharpen_13':{'port':9141, 'weight_path':'../trigger_gen/results/result_sharpen_13/sharpen_13.pkl'},
}

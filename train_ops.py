
class TrainOps:

    def populate(self, sess):
        graph = sess.graph
        self.train_d = graph.get_operation_by_name('train_d')
        self.train_g = graph.get_operation_by_name('train_g')
        self.loss_d = graph.get_tensor_by_name('loss/loss_d:0')
        self.loss_g = graph.get_tensor_by_name('loss/loss_g:0')
        self.generated_images = graph.get_tensor_by_name('generator/generated_images:0')
        self.global_step_var = graph.get_tensor_by_name('global_step:0')
        self.batch_var = graph.get_tensor_by_name('batch:0')
        self.epoch_var = graph.get_tensor_by_name('epoch:0')
        self.summary_op = graph.get_tensor_by_name('Merge/MergeSummary:0')  
        self.dx = graph.get_tensor_by_name('Dx:0')  
        self.dg = graph.get_tensor_by_name('Dg:0')

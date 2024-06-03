import json
import torch
from torch import nn
import numpy as np
import torch.optim as optim
from batch_embedding import batch_embed
from dataloader import workload_dataloader
import pdb

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class Network(nn.Module):
    def __init__(self, trees, file_name) -> None:
        super(Network, self).__init__()
        with open(file_name, "r") as f:
            col_id_map, col_stats, table_id_map, table_stats, type_id_map, op_stats, filter_op_id_map = json.load(f)

        target_runtime, batch_op_pad, batch_attr_pad, batch_filter_pad, batch_output_pad, batch_mapping_pad, batch_op, batch_attr, batch_filter, batch_output, batch_mapping = batch_embed(trees, 
                             max_filter=5, 
                             num_filter_attr=14, 
                             op_stats=op_stats, 
                             col_id_map=col_id_map,
                             col_stats=col_stats,
                             table_id_map=table_id_map,
                             table_stats=table_stats,
                             type_id_map=type_id_map,
                             filter_op_id_map=filter_op_id_map,
                             num_attr=4)
        
        labels = []
        for runtime in target_runtime:
            if runtime > 3000:
                labels.append(1)
            else:
                labels.append(0)
        labels = np.array(labels)

        self.col_id_map = col_id_map
        self.col_stats = col_stats
        self.table_id_map = table_id_map
        self.table_stats = table_stats
        self.type_id_map = type_id_map
        self.op_stats = op_stats
        self.filter_op_id_map = filter_op_id_map
        
        self.target_runtime = torch.tensor(target_runtime, dtype=torch.float32).to(device)
        self.labels = torch.tensor(labels, dtype=torch.float32).to(device)[:4000]
        self.hidden_dim = 128
        self.squeeze_dim = 32
        self.trees = trees[:4000]
        self.test = trees[4000:]
        self.test_labels = torch.tensor(labels, dtype=torch.float32).to(device)[4000:]
        
        self.filter_linear = nn.Linear(batch_filter_pad.shape[-1], self.squeeze_dim, dtype=torch.float32)
        self.output_linear = nn.Linear(batch_output_pad.shape[-1], self.squeeze_dim, dtype=torch.float32)
        self.batch_norm1 = nn.BatchNorm1d(self.squeeze_dim)
        self.batch_norm2 = nn.BatchNorm1d(self.squeeze_dim)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.lstm = nn.LSTM(input_size=2*self.squeeze_dim + batch_op_pad.shape[-1] + batch_attr_pad.shape[-1], hidden_size=self.hidden_dim, batch_first=True)

    def _adding_network(self):
        self.final_linear1 = nn.Linear(self.hidden_dim, 20, dtype=torch.float32).to(device)
        self.final_linear2 = nn.Linear(20, 1, dtype=torch.float32).to(device)

    def train(self, batch_size, num_epoch):
        self.batch_size = batch_size
        self._adding_network()
        self.optimizer = optim.SGD(self.parameters(), lr=0.01)
        num_batch = len(self.trees) / batch_size
        avg_loss = 0
        for epoch in range(num_epoch):
            print("epoch", epoch)
            for batch in range(int(num_batch)):
                labels = self.labels[batch_size*batch : batch_size*(batch+1)]
                runtime = self.target_runtime[batch_size*batch : batch_size*(batch+1)]
                sub_trees = self.trees[batch_size*batch : batch_size*(batch+1)]
                result = batch_embed(sub_trees, 
                             max_filter=5, 
                             num_filter_attr=14, 
                             op_stats=self.op_stats, 
                             col_id_map=self.col_id_map,
                             col_stats=self.col_stats,
                             table_id_map=self.table_id_map,
                             table_stats=self.table_stats,
                             type_id_map=self.type_id_map,
                             filter_op_id_map=self.filter_op_id_map,
                             num_attr=4)
                
                op_pad = torch.tensor(result[1], dtype=torch.float32).to(device)
                attr_pad = torch.tensor(result[2], dtype=torch.float32).to(device)
                filter_pad = torch.tensor(result[3], dtype=torch.float32).to(device)
                output_pad = torch.tensor(result[4], dtype=torch.float32).to(device)
                mapping_pad = result[5]

                output:torch.Tensor = self.forward(op_pad, attr_pad, filter_pad, output_pad, np.array(mapping_pad)).T[0]
                loss = self.Cross_Entropy(output, labels)
                # loss = self.Cross_Entropy(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # self.optimizer.zero_grad()
                
                avg_loss += loss
                if batch % 20 == 0:
                    print(output)
                    print(avg_loss/20)
                    avg_loss = 0

        return

    def forward(self, op_pad:torch.Tensor, attr_pad:torch.Tensor, filter_pad:torch.Tensor, output_pad:torch.Tensor, mapping_pad:np.ndarray):
        x_filter = filter_pad.shape[0]
        y_filter = filter_pad.shape[1]
        z_filter = filter_pad.shape[2]
        x_output = output_pad.shape[0]
        y_output = output_pad.shape[1]
        z_output = output_pad.shape[2]
        inter_filter_pad = filter_pad.view(x_filter*y_filter, z_filter)
        inter_output_pad = output_pad.view(x_output*y_output, z_output)
        filter_dense:torch.Tensor = self.relu(self.filter_linear(inter_filter_pad))
        filter_dense = self.batch_norm1(filter_dense)
        output_dense:torch.Tensor = self.relu(self.output_linear(inter_output_pad))
        output_dense = self.batch_norm2(output_dense)

        filter_pad = filter_dense.view(x_filter, y_filter, -1)
        output_pad = output_dense.view(x_output, y_output, -1)
        lstm_input = torch.cat((op_pad, attr_pad, filter_pad, output_pad), dim=2)
        num_layer = lstm_input.shape[0]
        inner_batch_size = lstm_input.shape[1]
        layer = num_layer
        h_t_1 = torch.zeros(1, inner_batch_size, self.hidden_dim).to(device)
        c_t_1 = torch.zeros(1, inner_batch_size, self.hidden_dim).to(device)
        
        while (layer > 0):
            layer_input = lstm_input[layer-1].view(inner_batch_size, 1, -1)
            layer_mapping = mapping_pad[layer-1]

            h_t = torch.zeros(1, inner_batch_size, self.hidden_dim).to(device)
            c_t = torch.zeros(1, inner_batch_size, self.hidden_dim).to(device)
            for i, mapping in enumerate(layer_mapping):
                if mapping[0] != 0:
                    h_t[0,i] += h_t_1[0, mapping[0]-1]
                    c_t[0,i] += c_t_1[0, mapping[0]-1]
                if mapping[1] != 0:
                    h_t[0,i] += h_t_1[0, mapping[1]-1]
                    c_t[0,i] += c_t_1[0, mapping[1]-1]
            c_t /= 2
            h_t /= 2
            _, (h_t_1, c_t_1) = self.lstm(layer_input, (h_t, c_t))

            layer -= 1

        output = h_t_1[0]
        output = output[:self.batch_size]
        output = self.final_linear1(output)
        output = self.final_linear2(output)
        output = self.sigmoid(output)
        
        return output
    
    def MSE_Loss(self, output, labels)-> torch.Tensor:
        return 1/2 * torch.mean((output - labels) ** 2)
    
    def Cross_Entropy(self, output, labels):
        bce_loss = - (labels * torch.log(output) + (1 - labels) * torch.log(1 - output))
        mean_loss = bce_loss.mean()
        return mean_loss
    
    def model_test(self):
        self.batch_size = 1
        self._adding_network()
        batch_size = 1
        test_set = self.test
        labels = self.test_labels
        num_batch = len(test_set) / batch_size
        out_list = []
        acc = 0

        for batch in range(int(num_batch)):
            sub_trees = self.trees[batch_size*batch : batch_size*(batch+1)]
            result = batch_embed(sub_trees, 
                            max_filter=5, 
                            num_filter_attr=14, 
                            op_stats=self.op_stats, 
                            col_id_map=self.col_id_map,
                            col_stats=self.col_stats,
                            table_id_map=self.table_id_map,
                            table_stats=self.table_stats,
                            type_id_map=self.type_id_map,
                            filter_op_id_map=self.filter_op_id_map,
                            num_attr=4)
            
            op_pad = torch.tensor(result[1], dtype=torch.float32).to(device)
            attr_pad = torch.tensor(result[2], dtype=torch.float32).to(device)
            filter_pad = torch.tensor(result[3], dtype=torch.float32).to(device)
            output_pad = torch.tensor(result[4], dtype=torch.float32).to(device)
            mapping_pad = result[5]

            output:torch.Tensor = self.forward(op_pad, attr_pad, filter_pad, output_pad, np.array(mapping_pad)).T[0]
            if output <= 0.5:
                out_list.append(0)
            else:
                out_list.append(1)
        
        for i in range (int(num_batch)):
            if out_list[i] == labels[i]:
                acc += 1

        return acc / num_batch

# Get the tree data.
filename = "../../../datasets/plans/parsed/workload_5k_s1_c8220.json"
dataloader = workload_dataloader(filename=filename)
trees = dataloader.get_data()

file_name = "../../../datasets/stats/stat_complete.json"
network = Network(trees=trees, file_name=file_name).to(device)
network.train(10, 50)
torch.save(network.state_dict(), 'model.pth')
acc = network.model_test()

print(acc)


# with open("../../../datasets/stats/stat_complete.json", "r") as f:
#     col_id_map, col_stats, table_id_map, table_stats, type_id_map, op_stats, filter_op_id_map = json.load(f)

# target_runtime, batch_op_pad, batch_attr_pad, batch_filter_pad, batch_output_pad, batch_mapping_pad, batch_op, batch_attr, batch_filter, batch_output, batch_mapping = batch_embed(trees[0:3], 
#                              max_filter=5, 
#                              num_filter_attr=14, 
#                              op_stats=op_stats, 
#                              col_id_map=col_id_map,
#                              col_stats=col_stats,
#                              table_id_map=table_id_map,
#                              table_stats=table_stats,
#                              type_id_map=type_id_map,
#                              filter_op_id_map=filter_op_id_map,
#                              num_attr=4)

# labels = []
# for runtime in target_runtime:
#     if runtime > 3000:
#         labels.append(1)
#     else:
#         labels.append(0)
# labels = np.array(labels)
# print(batch_op_pad.shape)
# print(batch_attr_pad.shape)
# print(batch_filter_pad.shape)
# print(batch_output_pad.shape)
# data = [batch_op_pad, batch_attr_pad, batch_filter_pad, batch_output_pad, batch_mapping_pad]
# len1 = batch_op_pad.shape[0]
# len2 = batch_op_pad.shape[1]
# len3 = batch_op_pad.shape[2]
# print(torch.tensor(batch_op_pad)[1])
# tensor1 = torch.tensor(batch_op_pad).view(len1*len2, -1)
# tensor2 = tensor1.view(len1, len2, len3)
# print(torch.equal(torch.tensor(batch_op_pad), tensor2))
# print(torch.tensor(batch_op_pad).view(len1*len2, -1)[:12])

# print(batch_mapping_pad[0:11])


    

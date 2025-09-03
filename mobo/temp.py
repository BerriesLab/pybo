# def save_dataset_to_csv(self, output_path: Path = None):
#
#     if output_path is None:
#         output_path = Path.cwd() / "dataset.csv"
#
#     XY = torch.cat([self._X, self._Y_obj], dim=-1)
#     if self._Y_obj_var is not None:
#         XY = torch.cat([XY, self._Y_obj_var], dim=-1)
#     if self._Y_con is not None:
#         XY = torch.cat([XY, self._Y_con], dim=-1)
#         if self._Y_con_var is not None:
#             XY = torch.cat([XY, self._Y_con_var], dim=-1)
#
#     XY = XY.detach().cpu().numpy()
#     np.savetxt(output_path, XY, delimiter=",", comments="")

# def to_json(self, output_path: Path = None):
#
#     if output_path is None:
#         output_path = Path.cwd() / "model.json"
#
#     serializable_data = {}
#     for key, value in self.__dict__.items():
#         json_key = key.lstrip('_')
#         serialized_value = serialize_value(value)
#         if serialized_value is not None:
#             serializable_data[json_key] = serialized_value
#
#     # Save to disk
#     with open(output_path, "w") as file:
#         json.dump(serializable_data, file, indent=4)

# def load_dataset_from_csv(
#         self,
#         input_space_dim: int | None = None,
#         objective_space_dim: int | None = None,
#         constraint_space_dim: int | None = None,
#         objective_variance: bool = False,
#         constraint_variance: bool = False,
#         filepath: str or None = None,
#         skiprows: int = 0,
#         skipcols: int = 0,
# ):
#     """Assumes that the dataset is saved in the CSV format and columns are ordered as follows:
#     X ¦ Y_obj ¦ Y_obj_var ¦ Y_con ¦ Y_con_var."""
#
#     if input_space_dim is None:
#         try:
#             # Get input dimensions from existing X tensor if available
#             input_space_dim = self._X.shape[-1]
#         except (AttributeError, RuntimeError, TypeError):
#             # X tensor isn't properly initialized or doesn't exist
#             raise ValueError(
#                 "Input space dimension must be provided explicitly as a parameter "
#                 "when X tensor is not initialized. Could not infer dimension from self._X."
#             )
#
#     if objective_space_dim is None:
#         try:
#             # Get objective dimensions from existing Y_obj tensor if available
#             objective_space_dim = self._Y_obj.shape[-1]
#         except (AttributeError, RuntimeError, TypeError):
#             # Y_obj tensor not properly initialized or doesn't exist
#             raise ValueError(
#                 "Objective space dimension must be provided explicitly as a parameter "
#                 "when Y_obj tensor is not initialized. Could not infer dimension from self._Y_obj."
#             )
#
#     if constraint_space_dim is None:
#         try:
#             constraints = self.get_output_constraints()
#             if constraints is not None and self._Y_con is not None:
#                 # The Problem is constrained and Y_con tensor exists
#                 constraint_space_dim = self._Y_con.shape[-1]
#             else:
#                 # The Problem is unconstrained or Y_con tensor doesn't exist
#                 constraint_space_dim = 0
#         except (AttributeError, RuntimeError, TypeError):
#             raise ValueError(
#                 "Constraint space dimension must be provided explicitly as a parameter "
#                 "since constraint tensor (Y_con) could not be determined automatically."
#             )
#
#     if filepath is None:
#         csv_files = list(Path("..").glob("*.csv"))
#         if not csv_files:
#             raise FileNotFoundError("No CSV files found in the current directory")
#         filepath = max(csv_files, key=lambda x: x.stat().st_mtime)
#
#     xy = np.loadtxt(filepath, delimiter=",", skiprows=skiprows)
#
#     idx = skipcols + 0
#     j = skipcols + input_space_dim
#     self._X = torch.tensor(xy[..., idx:j])
#
#     if objective_space_dim > 0:
#         idx = j
#         j += objective_space_dim
#         self._Y_obj = torch.tensor(xy[..., idx:j])
#
#         if objective_variance:
#             idx = j
#             j += objective_space_dim
#             self._Y_obj_var = torch.tensor(xy[..., idx:j])
#         else:
#             self._Y_obj_var = None
#     else:
#         self._Y_obj = None
#         self._Y_obj_var = None
#
#     if constraint_space_dim > 0:
#         idx = j
#         j += constraint_space_dim
#         self._Y_con = torch.tensor(xy[..., idx:j])
#
#         if constraint_variance:
#             idx = j
#             j += constraint_space_dim
#             self._Y_con_var = torch.tensor(xy[..., idx:j])
#         else:
#             self._Y_con_var = None
#     else:
#         self._Y_con = None
#         self._Y_con_var = None
